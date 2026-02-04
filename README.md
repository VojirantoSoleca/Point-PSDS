# Point-PSDS

## Learning from Geometric Redundancy: Spectral-Guided Masked Modeling for Self-Supervised Point Cloud Representation

Self-supervised learning (SSL) for point clouds aims to alleviate the dependency on large-scale annotated datasets. Recently, masked modeling has become a dominant paradigm; however, the prevalent random masking strategy ignores intrinsic geometric structures, inefficiently treating complex edges and flat surfaces with equal importance. To address this, we propose Point-PSDS, a novel geometry-aware framework. We introduce Patch Spectral Diffusion Stability (PSDS), a metric utilizing graph spectral diffusion to quantify the structural redundancy of local patches. Distinct from existing methods that mask salient features, we strategically mask the most geometrically redundant regions (i.e., those with the lowest PSDS scores). This strategy compels the network to reconstruct simple surfaces by understanding their surrounding complex structural context, effectively learning robust geometric priors. Furthermore, we integrate a Cross-View Distillation loss within a teacher-student architecture to enhance the view-invariance and semantic consistency of the learned features. Extensive experiments demonstrate that our method achieves state-of-the-art results compared to models of comparable size, establishing new benchmarks on ScanObjectNN and few-shot learning, while maintaining comparable top-tier performance on ModelNet40.

<div  align="center">    
 <img src="./figure/net.jpg" width = "666"  align=center />
</div>

## 1. Requirements
PyTorch >= 1.7.0 < 1.11.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
pip install -r requirements.txt
```
<details>
<summary> For Linux Kernel 6.0 or above (e.g. Ubuntu 24)
</summary>

Please run the following command before installing Chamfer Distance:
```
sudo apt install gcc-10 g++-10

su
cd /usr/local/src
wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
tar -xf linux-5.4.tar.xz && cd linux-5.4
make headers_install INSTALL_HDR_PATH=/usr/local/linux-headers-5.4

export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CFLAGS="-I/usr/local/linux-headers-5.4/include"
export CPPFLAGS="-I/usr/local/linux-headers-5.4/include"
```

In `extensions/chamfer_dist/setup.py`, in the `extra_compile_args` field, pass the correct header path to nvcc by adding the following line as the second element of `ext_modules`:
```
extra_compile_args={"nvcc": ['--system-include=/usr/local/linux-headers-5.4/include']}
```

</details>

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. Pre-training
To pretrain Point-PSDS on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
```
## 4. Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

<div  align="center">    
 <img src="./figure/vvv.jpg" width = "900"  align=center />
</div>
  year={2022},
  organization={Springer}
}
```
