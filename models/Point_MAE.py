import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from torchvision import transforms
from datasets import data_transforms
from copy import deepcopy
import copy
import math
# from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class TwoViewPointcloudTransform:
    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, pointcloud):
        u = copy.deepcopy(pointcloud)
        u = self.base_transforms(u)


        v = copy.deepcopy(pointcloud)
        v = self.base_transforms(v)

        return u, v

def knn(x, y=None, k=32):
    """
    x: [B, N, C] - all points
    y: [B, M, C] - center points
    return: dists [B, M, k], idx [B, M, k]
    """
    if y is None:
        y = x

    B, N, C = x.shape
    _, M, _ = y.shape

    x_square = torch.sum(y ** 2, dim=-1, keepdim=True)  # [B, M, 1]
    y_square = torch.sum(x ** 2, dim=-1).unsqueeze(1)  # [B, 1, N]
    inner = torch.bmm(y, x.transpose(1, 2))  # [B, M, N]

    dists = x_square + y_square - 2 * inner  # [B, M, N]
    dists, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)
    return dists, idx


class Encoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = knn(xyz, center, k=self.group_size)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # self.mask_type = config.transformer_config.mask_type
        self.mask_type = "curvature"  # 'block', 'rand', 'curvature'

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def _mask_center_curvature(self, center, curvature_scores, noaug=False, mask_type=0.5, mask_value=0):
        '''
            center: [B, G, 3]
            curvature_scores: [B, G] 
                mask: [B, G]
        '''
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2], dtype=torch.bool, device=center.device)

        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)

        if mask_type < 0.5:
            value = [0.6, 0.7]
            num_high = int(value[mask_value] * G)
            _, high_idx = torch.topk(curvature_scores, k=num_high, dim=1, largest=True)
            for b in range(B):
                mask[b, high_idx[b]] = True
        else:
            value = [0.6, 0.7]
            num_low = int(value[mask_value] * G)
            _, low_idx = torch.topk(curvature_scores, k=num_low, dim=1, largest=False)
            for b in range(B):
                mask[b, low_idx[b]] = True

        return mask

    def compute_laplacian_min_scores(self, neighborhood, normalize=True, max_iter=50, tol=1e-6, eps=1e-6):
        B, G, M, _ = neighborhood.shape
        device = neighborhood.device

        patches_flat = neighborhood.reshape(B * G, M, 3)

        dist = torch.cdist(patches_flat, patches_flat)
        min_dist = torch.min(dist.masked_fill(dist == 0, float('inf')), dim=-1, keepdim=True).values
        adj = 1.0 / (dist / (min_dist + eps) + torch.eye(M, device=device).unsqueeze(0))

        if normalize:
            D = torch.sum(adj, dim=-1)
            D_inv_sqrt = torch.rsqrt(D + eps)
            D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
            L = torch.eye(M, device=device).unsqueeze(0) - D_inv_sqrt @ adj @ D_inv_sqrt
        else:
            D = torch.diag_embed(torch.sum(adj, dim=-1))
            L = D - adj

        I = torch.eye(M, device=device).unsqueeze(0)
        M_mat = I - L

        v = torch.randn_like(M_mat[:, :, 0])
        v = v / (v.norm(dim=-1, keepdim=True) + eps)

        for _ in range(max_iter):
            v_next = torch.bmm(M_mat, v.unsqueeze(-1)).squeeze(-1)
            v_next = v_next / (v_next.norm(dim=-1, keepdim=True) + eps)
            if torch.max(torch.abs(v_next - v)) < tol:
                break
            v = v_next

        v = v.unsqueeze(-1)
        lambda_min = torch.bmm(torch.bmm(v.transpose(1, 2), L), v).squeeze(-1).squeeze(-1)

        min_eig_scores = lambda_min.reshape(B, G)
        return min_eig_scores

    def compute_curvature_scores(self, neighborhood):
        """
        neighborhood: [B, G, M, 3]
            curvature_scores: [B, G]
        """
        B, G, M, _ = neighborhood.shape
        pts_centered = neighborhood - neighborhood.mean(dim=2, keepdim=True)

        pts_centered_t = pts_centered.transpose(2, 3)  # (B, G, 3, M)
        cov = torch.matmul(pts_centered_t, pts_centered) / (M - 1 + 1e-6)  # (B, G, 3, 3)

        eigvals = torch.linalg.eigvalsh(cov)

        eigvals = torch.clamp(eigvals, min=1e-12)

        curvature = eigvals[..., 0] / eigvals.sum(dim=-1)

        return curvature

    def forward(self, neighborhood, center, noaug=False, no_Mask=False, mask_type=0.5, mask_value=0):
        if no_Mask:
            group_input_tokens = self.encoder(neighborhood)  # B G C
            batch_size, seq_len, C = group_input_tokens.size()
            pos = self.pos_embed(center)
            x_vis = self.blocks(group_input_tokens, pos)
            x_vis = self.norm(x_vis)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=neighborhood.device)
            return x_vis, mask

        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        elif self.mask_type == 'curvature':
            curvature_scores = self.compute_laplacian_min_scores(neighborhood)
            bool_masked_pos = self._mask_center_curvature(center, curvature_scores, noaug=noaug, mask_type=mask_type, mask_value=mask_value)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos

class MLPHead(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=512, out_dim=512, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm_last_layer = norm_last_layer
        if norm_last_layer:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]

        if x.dim() == 3:
            B, N, C = x.shape
            x = x.view(B * N, C)
            x = self.mlp(x)
            if self.norm_last_layer:
                x = self.ln(x)
            x = x.view(B, N, -1)
        elif x.dim() == 2:
            x = self.mlp(x)
            if self.norm_last_layer:
                x = self.ln(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return x

def get_cosine_temp(start_temp, end_temp, cur_epoch, max_epoch):
    ratio = cur_epoch / max_epoch
    return end_temp + 0.5 * (start_temp - end_temp) * (1 + math.cos(math.pi * ratio))

def teacher_confidence_stats_512(t_out_mask, name="view"):
    B, M, D = t_out_mask.shape
    assert D == 512, f"Expected distill_dim=512, got {D}"

    max_prob = t_out_mask.max(dim=-1)[0]          # [B, M]
    mean_max_prob = max_prob.mean().item()

    entropy = -(t_out_mask * (t_out_mask + 1e-8).log()).sum(dim=-1)  # [B, M]
    mean_entropy = entropy.mean().item()

    max_entropy = np.log(512)
    if mean_max_prob > 0.95 or mean_entropy < 0.1:
        print(f"[Warning] Teacher {name} may be over-confident!")
    if mean_max_prob < 0.05 or mean_entropy > max_entropy * 0.9: 
        print(f"[Warning] Teacher {name} may be under-confident!")

@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        self.lambda_distill = 1.0 
        self.teacher_encoder = None 

        self.teacher_encoder = deepcopy(self.MAE_encoder)  # freeze or update by EMA
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_head = MLPHead(in_dim=384, out_dim=512)  # D_out = distill_dim
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        self.student_head = MLPHead(in_dim=384, out_dim=512)
        self.teacher_temp = 1.0
        self.student_temp = 1.0
        self.alpha = 0.001

        base_transforms = transforms.Compose(
            [
                data_transforms.PointcloudScaleAndTranslate(),
                data_transforms.PointcloudRotate(),
                data_transforms.PointcloudJitter(),
            ]
        )
        self.train_transforms = TwoViewPointcloudTransform(base_transforms)
        self.register_buffer("running_recon", torch.tensor(1.0))
        self.register_buffer("running_distill", torch.tensor(1.0))
        self.momentum = 0.9 
        self.teacher_initialized = False

    @torch.no_grad()
    def update_teacher(self, m=0.996):
        for param_q, param_k in zip(self.MAE_encoder.parameters(), self.teacher_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)


    @torch.no_grad()
    def update_temp(self, epoch, max_epoch, t_out_mask_u_logits, t_out_mask_v_logits):
        if not hasattr(self, 'teacher_temp') or self.teacher_temp is None:
            self.teacher_temp = 0.7

        cur_temp = float(self.teacher_temp) 

        p_u = F.softmax(t_out_mask_u_logits / cur_temp, dim=-1)
        p_v = F.softmax(t_out_mask_v_logits / cur_temp, dim=-1)

        eps = 1e-8
        p_u = p_u.clamp(min=eps)
        p_v = p_v.clamp(min=eps)

        entropy_u = -(p_u * p_u.log()).sum(dim=-1)  # [B, M]
        entropy_v = -(p_v * p_v.log()).sum(dim=-1)
        mean_entropy_u = float(entropy_u.mean())
        mean_entropy_v = float(entropy_v.mean())
        mean_entropy = 0.5 * (mean_entropy_u + mean_entropy_v)

        D = t_out_mask_u_logits.shape[-1]
        max_entropy = math.log(D)
        target_entropy = max_entropy * 0.6

        alpha = 0.01
        delta = alpha * (target_entropy - mean_entropy)

        new_temp = float(self.teacher_temp + delta)
        new_temp = max(0.7, min(new_temp, 1.5))
        self.teacher_temp = new_temp

        self.student_temp = min(1.0, self.teacher_temp)
        self.student_temp = max(self.student_temp, 0.7)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, epoch=0, **kwargs):
        # ======== Step 1: frequency-masked view for student ========
        u, v = self.train_transforms(pts)

        neighborhood_u, center_u = self.group_divider(u)  # B G M 3, B G 3
        neighborhood_v, center_v = self.group_divider(v)  # B G M 3, B G 3

        with torch.no_grad():
            t_neighborhood_u, t_center_u = neighborhood_u.clone().detach(), center_u.clone().detach()
            t_neighborhood_v, t_center_v = neighborhood_v.clone().detach(), center_v.clone().detach()

        mask_type = 0.9

        x_vis_u, mask_u = self.MAE_encoder(neighborhood_u, center_u, mask_type=mask_type, mask_value=0)  # B VIS C
        x_vis_v, mask_v = self.MAE_encoder(neighborhood_v, center_v, mask_type=mask_type, mask_value=0)  # B VIS C

        B_u, M_u, C_u = x_vis_u.shape
        pos_emd_vis_u = self.decoder_pos_embed(center_u[~mask_u]).reshape(B_u, -1, C_u)
        pos_emd_mask_u = self.decoder_pos_embed(center_u[mask_u]).reshape(B_u, -1, C_u)
        B_v, M_v, C_v = x_vis_v.shape
        pos_emd_vis_v = self.decoder_pos_embed(center_v[~mask_v]).reshape(B_v, -1, C_v)
        pos_emd_mask_v = self.decoder_pos_embed(center_v[mask_v]).reshape(B_v, -1, C_v)

        _, N_u, _ = pos_emd_mask_u.shape
        mask_token_u = self.mask_token.expand(B_u, N_u, -1)
        x_full_u = torch.cat([x_vis_u, mask_token_u], dim=1)
        pos_full_u = torch.cat([pos_emd_vis_u, pos_emd_mask_u], dim=1)
        _, N_v, _ = pos_emd_mask_v.shape
        mask_token_v = self.mask_token.expand(B_v, N_v, -1)
        x_full_v = torch.cat([x_vis_v, mask_token_v], dim=1)
        pos_full_v = torch.cat([pos_emd_vis_v, pos_emd_mask_v], dim=1)

        x_rec_u = self.MAE_decoder(x_full_u, pos_full_u, N_u)  # B M C
        rebuild_points_u = self.increase_dim(x_rec_u.transpose(1, 2)).transpose(1, 2).reshape(B_u * N_u, -1, 3)
        gt_points_u = neighborhood_u[mask_u].reshape(B_u * N_u, -1, 3)
        loss_mfm_u = self.loss_func(rebuild_points_u, gt_points_u)
        x_rec_v = self.MAE_decoder(x_full_v, pos_full_v, N_v)  # B M C
        rebuild_points_v = self.increase_dim(x_rec_v.transpose(1, 2)).transpose(1, 2).reshape(B_v * N_v, -1, 3)
        gt_points_v = neighborhood_v[mask_v].reshape(B_v * N_v, -1, 3)
        loss_mfm_v = self.loss_func(rebuild_points_v, gt_points_v)

        loss_mfm = loss_mfm_u + loss_mfm_v

        # ======== Step 2: teacher branch - original input no masking ========
        if self.teacher_initialized and epoch > 60:
            with torch.no_grad():
                # Freeze teacher
                # teacher for view u
                # t_neighborhood_u, t_center_u = self.group_divider(u)  # B G M 3, B G 3
                t_feat_u = self.teacher_encoder(t_neighborhood_u, t_center_u, no_Mask=True)  # (B, N, C)
                t_out_u = self.teacher_head(t_feat_u)  # (B, N, D) logits

                # view v
                t_feat_v = self.teacher_encoder(t_neighborhood_v, t_center_v, no_Mask=True)  # (B, N, C)
                t_out_v = self.teacher_head(t_feat_v)  # (B, N, D) logits

                # ====== Select teacher logits corresponding to student masked tokens ======
                # view u
                B_u, N_u, D = t_out_u.shape
                mask_u_flat = mask_u.reshape(B_u * N_u)
                t_out_u_flat = t_out_u.reshape(B_u * N_u, D)
                M_u = mask_u.sum(dim=1)[0].item()
                t_out_mask_u = t_out_u_flat[mask_u_flat].reshape(B_u, M_u, D)  # (B, M, D)

                # view v
                B_v, N_v, D = t_out_v.shape
                mask_v_flat = mask_v.reshape(B_v * N_v)
                t_out_v_flat = t_out_v.reshape(B_v * N_v, D)
                M_v = mask_v.sum(dim=1)[0].item()
                t_out_mask_v = t_out_v_flat[mask_v_flat].reshape(B_v, M_v, D)  # (B, M, D)

                # ====== Update teacher temperature based on logits ======
                self.update_temp(epoch, 300, t_out_mask_u, t_out_mask_v)

                # ====== Apply softmax with updated teacher_temp ======
                t_out_mask_u = F.softmax(t_out_mask_u / self.teacher_temp, dim=-1).detach()
                t_out_mask_v = F.softmax(t_out_mask_v / self.teacher_temp, dim=-1).detach()

            # ======== Step 3: student head prediction from masked input ========
            s_feat_u = self.student_head(x_rec_u)  # (B, masked_token_num, D)
            s_out_u = F.log_softmax(s_feat_u / self.student_temp, dim=-1)
            s_feat_v = self.student_head(x_rec_v)
            s_out_v = F.log_softmax(s_feat_v / self.student_temp, dim=-1)

            kl_u_per_token = F.kl_div(s_out_u, t_out_mask_v.detach(), reduction='none')  # (B, M, D)
            kl_u_mean = kl_u_per_token.mean(dim=-1)  # (B, M)

            kl_v_per_token = F.kl_div(s_out_v, t_out_mask_u.detach(), reduction='none')
            kl_v_mean = kl_v_per_token.mean(dim=-1)

            loss_dis = (kl_u_mean + kl_v_mean) * (self.student_temp * self.teacher_temp)
            loss_dis = loss_dis.clamp(min=1e-6)

            return loss_mfm, loss_dis

        else:
            return loss_mfm, loss_mfm


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret