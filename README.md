# OpenVid
Text-to-video (T2V) generation has recently garnered significant attention, largely due to the advanced multi-modality model, Sora. However, current T2V generation in research community still faces two major challenges: 1) The absence of a precise, high-quality open-source dataset. Previous popular video datasets, such as WebVid-10M and Panda-70M, are either of low quality or too large for most research institutions. Collecting precise, high-quality text-video pairs is both challenging and essential for T2V generation. 2) Inadequate utilization of textual information. Recent T2V methods focus on vision transformers, employing a simple cross-attention module for video generation, which falls short of thoroughly extracting semantic information from text prompt.
To address these issues, we introduce OpenVid-1M, a high-quality dataset with expressive captions. This open-scenario dataset comprises over 1 million text-video pairs, facilitating T2V generation research. Additionally, we curate 433K 1080p videos from OpenVid-1M to create OpenVidHD-0.4M, advancing high-definition video generation. Furthermore, we propose a novel Multi-modal Video Diffusion Transformer (MVDiT), capable of extracting structural information from visual tokens and semantic information from text tokens. Extensive experiments and ablation studies demonstrate the superiority of OpenVid-1M over previous datasets and the effectiveness of our MVDiT.

# Environment

```bash
conda create -n openvid python=3.10
conda activate openvid
pip install torch torchvision
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

# Usage
## Preparation

### Dataset
1. Download [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) dataset.
2. Put the dataset in `./dataset` folder.
```
dataset
└─ OpenVid-1M
    └─ data
        └─ train
            └─ OpenVid-1M.csv
            └─ OpenVidHD.csv
    └─ video
```

### Model Weight
| Model | Data | Steps | Batch Size | URL                                                                                           |
|------------|--------|-------------|------------|-----------------------------------------------------------------------------------------------|
| STDiT-16×1024×1024 | OpenVidHQ | 16k | 32×4 | [:link:](https://huggingface.co/datasets/nkp37/OpenVid-1M) |
| STDiT-16×512×512 | OpenVid-1M | 20k | 32×8 | [:link:](https://huggingface.co/datasets/nkp37/OpenVid-1M) |
| MVDiT-16×512×512 | OpenVid-1M | 20k | 32×4 | [:link:](https://huggingface.co/datasets/nkp37/OpenVid-1M) |

Training orders: 16x256x256 $\rightarrow$ 16×512×512 $\rightarrow$ 16×1024×1024.

Our model's weight is partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha).

# Inference
```bash
# MVDiT 16x512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/mvdit/inference/16x512x512.py --ckpt-path MVDiT-16x512x512.pth

# STDiT 16x512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/stdit/inference/16x512x512.py --ckpt-path STDiT-16x512x512.pth

# STDiT 16x1024x1024
torchrun --standalone --nproc_per_node 1 scripts/inference.py configs/stdit/inference/16x1024x1024.py --ckpt-path STDiT-16x1024x1024.pth 
```

# Training
```bash
# MVDiT 16x512x512
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/mvdit/train/16x512x512.py

# STDiT 16x512x512
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py configs/stdit/train/16x512x512.py
```

# References
Part of the code is based upon:
[Open-Sora](https://github.com/hpcaitech/Open-Sora).
Thanks for their great work!