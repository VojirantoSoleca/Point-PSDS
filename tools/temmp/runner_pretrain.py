import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from copy import deepcopy
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
import math
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.module.'):
            new_k = k[len('module.module.'):]
        elif k.startswith('module.'):
            new_k = k[len('module.'):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get(
        'extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    chechpoint = None
    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)

    if args.start_ckpts is not None:
        # builder.load_model(base_model, args.start_ckpts, logger=logger)
        checkpoint = torch.load(args.start_ckpts, map_location='cpu')

        # 加载 base_model

        # 加载 teacher_encoder（注意：不是整个 teacher_model，只是 encoder）
        cleaned_base_model_ckpt = strip_module_prefix(checkpoint['base_model'])
        base_model.load_state_dict(cleaned_base_model_ckpt)
        cleaned_teacher_encoder = strip_module_prefix(checkpoint['teacher_encoder'])
        base_model.teacher_encoder.load_state_dict(cleaned_teacher_encoder)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    if args.start_ckpts is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        # if start_epoch > 60:
        #     base_model.module.teacher_initialized = True
        # if start_epoch < 60:
        #     base_model.module.teacher_temp =  0.07
        # elif start_epoch < 90:
        #     base_model.module.teacher_temp = 0.1
        # elif start_epoch < 150:
        #     base_model.module.teacher_temp = 0.3
        # else:
        #     base_model.module.teacher_temp = 0.5
        # base_model.module.student_temp = base_model.module.teacher_temp * 0.8
        # base_model.module.update_temp(start_epoch, config.max_epoch + 1)
        if start_epoch > -1:
            base_model.module.teacher_initialized = True

    print("==== Optimizer param groups ====")
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}, weight_decay={group['weight_decay']}, num_params={len(group['params'])}")
        for p in group['params']:
            # 找到 param 的名字
            for name, param in base_model.module.named_parameters():
                if p is param:
                    print(f"  {name}")
                    break

    # trainval
    # training
    base_model.zero_grad()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        def get_momentum(epoch):
            # warmup 到 0.9995
            return min(0.996 + (epoch / config.max_epoch) * (0.9995 - 0.996), 0.9995)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints

            if epoch == 0 and base_model.module.teacher_initialized is False:
                base_model.module.teacher_initialized = True
                # base_model.module.teacher_encoder.load_state_dict(base_model.module.MAE_encoder.state_dict())
                # base_model.module.update_temp(epoch, config.max_epoch + 1)

            # if epoch == 181 and base_model.module.teacher_initialized is True:
            #     base_model.module.teacher_initialized = False

            # points = train_transforms(points)
            l1, l2 = base_model(points, epoch=epoch)

            def get_lambda(epoch, warmup=30, lambda_max=1, decay_after=None):
                """
                - warmup: epochs to linearly increase from 0 to lambda_max
                - lambda_max: target maximum weight (建议 0.01)
                - decay_after: if not None, epoch after which linearly decay to lambda_max*0.5 by end
                """
                if epoch < warmup:
                    return 0
                else :
                    return 1

            # if epoch < 30:
            #     loss = l1 / 2
            # else:
            #     if sc < 100:
            #         loss = (l1 + l2 * sc * get_lambda(epoch)) / 2
            #     else:
            #         loss = (l1 + l2 * 100 * get_lambda(epoch)) / (1 + 100 / sc)
            #
            #     loss = loss / 2

            def get_distill_weight(epoch, start_epoch=60, max_epoch=300, weight_start=0.1, weight_max=0.25):
                """
                根据 epoch 计算蒸馏权重。

                Args:
                    epoch (int): 当前训练 epoch
                    start_epoch (int): 开始蒸馏的 epoch
                    max_epoch (int): 总训练 epoch
                    weight_start (float): 蒸馏初始权重
                    weight_max (float): 蒸馏最大权重

                Returns:
                    float: 当前 epoch 的蒸馏权重
                """
                if epoch < start_epoch:
                    return 0.0  # 蒸馏未开始
                else:
                    # 线性增长到 weight_max
                    progress = (epoch - start_epoch) / (max_epoch - start_epoch)
                    return weight_start + progress * (weight_max - weight_start)

            if epoch <= 60 or epoch > 180:
                loss = l1 * 0.5
            else:
                # w = (sc / (sc + 1.0)).detach()
                # loss = w * l1 * 0.5 + (1.0 - w) * l2 * 0.5
                loss = (l1 + l2 * 0.1) * 0.5

            #
            # loss = l1 + l2 * get_lambda(epoch, config.max_epoch)
            #
            # loss *= 0.25

            if torch.isnan(loss).any():
                print(f"[Error] NaN loss detected at epoch {epoch}, batch {idx}. Terminating training.")
                if torch.isnan(l1):
                    print(f"[Error] NaN l1.")
                if torch.isnan(l2):
                    print(f"[Error] NaN l2.")
                exit(1)  # 或 raise RuntimeError("NaN loss encountered.")

            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0

                optimizer.step()
                base_model.zero_grad()

                if epoch > -1:
                    # EMA update (Teacher <- Student)
                    def get_module(m):
                        return m.module if hasattr(m, 'module') else m

                    base_model.module.update_teacher(m=get_momentum(epoch))

                    # base_model.module.update_temp(epoch, config.max_epoch + 1)

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 50 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s loss_recon = %s loss_distill = %s lr = %.6f tt = %s st = %s' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], l1.mean().item() * 1000, l2.mean().item() * 1000, optimizer.param_groups[0]['lr'], base_model.module.teacher_temp, base_model.module.student_temp), logger=logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, scheduler, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if epoch % 20 == 0 or epoch == 30:
            builder.save_checkpoint(base_model, optimizer, scheduler, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                               test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass