<p align="center">
  <img src="assets/logo.jpg"  height=360>
</p>


### <div align="center"> OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation <div> 
<div align="center">
  <a href="https://nju-pcalab.github.io/projects/openvid/"><img src="https://img.shields.io/static/v1?label=OpenVid-1M&message=Project&color=purple"></a> &ensp;
  <a href="http://export.arxiv.org/pdf/2407.02371"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/datasets/nkp37/OpenVid-1M"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow"></a> &ensp;
</div>


## OpenVid-1M
OpenVid-1M is a high-quality text-to-video dataset designed for research institutions to enhance **video quality, featuring high aesthetics, clarity, and resolution**. It can be used for direct training or as a quality tuning complement to other video datasets. It can also be used in other video generation task (video super-resolution, frame interpolation, etc)

We carefully curate 1 million high-quality video clips with expressive captions to advance text-to-video research, in which **0.4 million videos are in 1080P resolution (termed OpenVidHD-0.4M)**.

OpenVid-1M is cited, discussed or used in several recent works, including video diffusion models [**MarDini**](https://arxiv.org/pdf/2410.20280), [**Allegro**](https://github.com/rhymes-ai/Allegro), [**T2V-Turbo-V2**](https://t2v-turbo-v2.github.io/), [**Pyramid Flow**](https://pyramid-flow.github.io/), [**SnapGen-V**](https://arxiv.org/pdf/2412.10494); long video generation model with AR model [**ARLON**](https://arxiv.org/pdf/2410.20502); visual understanding and generation model [**VILA-U**](https://arxiv.org/pdf/2409.04429); 3D/4D generation models [**GenXD**](https://arxiv.org/pdf/2411.02319), [**DimentionX**](https://arxiv.org/pdf/2411.04928?); video VAE model [**IV-VAE**](https://arxiv.org/pdf/2411.06449); Frame interpolation model [**Framer**](https://openreview.net/pdf?id=Lp40Z40N07) and large multimodal model [**InternVL 2.5**](https://arxiv.org/pdf/2412.05271).

## News 🚀🚀🚀
- **[2025.01.23]** 🏆 OpenVid-1M is accepted by **ICLR 2025**!!!
- **[2024.12.01]** 🚀 OpenVid-1M dataset was downloaded over **79,000** times on Huggingface last month, placing it in the **top 1%** of all video datasets (as of Nov. 2024)!!
- **[2024.07.01]** 🔥 Our paper, code, model and OpenVid-1M dataset are released!

## Preparation
### Environment
```bash
conda create -n openvid python=3.10
conda activate openvid
pip install torch torchvision
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Dataset
1. Download [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) dataset.
```bash
# it takes a lot of time.
python download_scripts/download_OpenVid.py
```
2. Put OpenVid-1M dataset in `./dataset` folder.
```
dataset
└─ OpenVid-1M
    └─ data
        └─ train
            └─ OpenVid-1M.csv
            └─ OpenVidHD.csv
    └─ video
        └─ ---_iRTHryQ_13_0to241.mp4
        └─ ---agFLYkbY_7_0to303.mp4
        └─ --0ETtekpw0_2_18to486.mp4
        └─ ...
```

### Model Weight
| Model | Data | Pretrained Weight | Steps | Batch Size | URL                                                                                           |
|------------|--------|--------|-------------|------------|-----------------------------------------------------------------------------------------------|
| STDiT-16×1024×1024 | OpenVidHQ | STDiT-16×512×512 | 16k | 32×4 | [:link:](https://huggingface.co/nkp37/OpenVid-1M/tree/main/model_weights) |
| STDiT-16×512×512 | OpenVid-1M | STDiT-16×256×256 | 20k | 32×8 | [:link:](https://huggingface.co/nkp37/OpenVid-1M/tree/main/model_weights) |
| MVDiT-16×512×512 | OpenVid-1M | MVDiT-16×256×256 | 20k | 32×4 | [:link:](https://huggingface.co/nkp37/OpenVid-1M/tree/main/model_weights) |

Our model's weight is partially initialized from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha).

## Inference
```bash
# MVDiT, 16x512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py --config configs/mvdit/inference/16x512x512.py --ckpt-path MVDiT-16x512x512.pt
# STDiT, 16x512x512
torchrun --standalone --nproc_per_node 1 scripts/inference.py --config configs/stdit/inference/16x512x512.py --ckpt-path STDiT-16x512x512.pt
# STDiT, 16x1024x1024
torchrun --standalone --nproc_per_node 1 scripts/inference.py --config configs/stdit/inference/16x1024x1024.py --ckpt-path STDiT-16x1024x1024.pt
```

## Training
```bash
# MVDiT, 16x256x256, 72k Steps
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py --config configs/mvdit/train/16x256x256.py
# MVDiT, 16x512x512, 20k Steps
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py --config configs/mvdit/train/16x512x512.py

# STDiT, 16x256x256, 72k Steps
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py --config configs/stdit/train/16x256x256.py
# STDiT, 16x512x512, 20k Steps
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py --config configs/stdit/train/16x512x512.py
# STDiT, 16x1024x1024, 16k Steps
torchrun --nnodes=1 --nproc_per_node=1 scripts/train.py --config configs/stdit/train/16x1024x1024.py
```
Training orders: 16x256x256 $\rightarrow$ 16×512×512 $\rightarrow$ 16×1024×1024.

## References
Part of the code is based upon:
[Open-Sora](https://github.com/hpcaitech/Open-Sora).
Thanks for their great work!

## Citation
```bibtex
@article{nan2024openvid,
  title={OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation},
  author={Nan, Kepan and Xie, Rui and Zhou, Penghao and Fan, Tiehan and Yang, Zhenheng and Chen, Zhijie and Li, Xiang and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2407.02371},
  year={2024}
}
```