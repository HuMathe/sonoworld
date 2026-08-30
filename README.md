# SonoWorld: From One Image to a 3D Audio-Visual Scene (CVPR 2026)

> Official implementation of the CVPR 2026 paper _"SonoWorld: From One Image to a 3D Audio-Visual Scene."_
> 

[![Paper](https://img.shields.io/badge/arXiv-2603.28757-CC3355?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=333333)](https://arxiv.org/abs/2603.28757)
[![Project Website](https://img.shields.io/badge/Project-Website-3E7CB1?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=333333)](https://humathe.github.io/sonoworld/)
[![Demo](https://img.shields.io/badge/Demo-Try%20it-4CAF7D?style=for-the-badge&logo=rocket&logoColor=white&labelColor=333333)](https://humathe.github.io/sonoworld/demo.html)
[![Hugging Face](https://img.shields.io/badge/Dataset-SonoScene360-F2B84B?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=333333)](https://huggingface.co/datasets/DerongJin/SonoScene360)

**TL;DR** Given a single input image, SonoWorld generates a 3D audio-visual scene with spatialized sound and scene-level assets.

<span style="color:#FF7A1A">(New)</span> The SonoScene360 dataset is now available at [https://huggingface.co/datasets/DerongJin/SonoScene360](https://huggingface.co/datasets/DerongJin/SonoScene360).


## Quick Start
After installing the required dependencies and preparing any model credentials/checkpoints needed by the selected stages, run:

```bash
python generate.py \
    --scene_root outputs/example_scene \
    --config configs/default.yaml \
    --input_image test-inputs/fall.jpg
```

Use `--resume` to continue a partially completed scene and `--force` to rerun completed stages.

### Panorama input

Use a complete 2:1 equirectangular panorama as input:

```bash
python generate.py \
    --scene_root outputs/fountain-multi \
    --config configs/default.yaml \
    --input_panorama /path/to/panorama.jpg
```

## Evaluation on SonoScene360 Dataset

### 1. Download

```bash
hf download DerongJin/SonoScene360 \
    --repo-type dataset \
    --local-dir SonoScene360
```

### 2. Generate

```bash
# Set SONOSCENE360_ROOT in generate_all_sonoscene360.sh, then run:
bash generate_all_sonoscene360.sh
```

### 3. Evaluate

```bash
# Set SONOSCENE360_ROOT in eval_sonoscene360.sh, then run:
bash eval_sonoscene360.sh
```

## News and Planned TODOs

- [x] `08.19.2026` Released [SonoScene360 dataset](https://huggingface.co/datasets/DerongJin/SonoScene360)
- [x] `06.17.2026` Environment setup instructions
- [x] `06.02.2026` Released generation code
- [ ] Rendering code, evaluation tools, interactive viewer, and additional examples
- [ ] Support for generation with open-source backbones, specifically HunyuanWorld 1.0 and LLaVA

## Installation

We tested SonoWorld on an NVIDIA A6000 with GCC 14.2.0, CUDA 12.4.1, and Python 3.12.

Clone the repository and create the environment:
```bash
git clone --branch main --single-branch git@github.com:HuMathe/sonoworld.git
cd sonoworld
conda create -n sonoworld python=3.12
conda activate sonoworld
```

Install PyTorch
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

Install [SAM3](https://github.com/facebookresearch/sam3), [MMAudio](https://github.com/hkchengrex/MMAudio), [GeoCalib](https://github.com/cvg/GeoCalib), and [MoGe](https://github.com/microsoft/MoGe):
```bash
mkdir -p third_party

git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 757bbb0206a0b68bee81b17d7eb4877177025b2f
pip install -e third_party/sam3

git clone https://github.com/hkchengrex/MMAudio.git third_party/MMAudio
pip install -e third_party/MMAudio

git clone https://github.com/cvg/GeoCalib.git third_party/GeoCalib
pip install -e third_party/GeoCalib

pip install git+https://github.com/microsoft/MoGe.git
```

Install the remaining dependencies:
```bash
pip install -r requirements.txt
pip install --force-reinstall "setuptools<82"
conda install -c conda-forge ffmpeg
```

Export your OpenAI api_key (to use GPT-5 for sounding category proposal):
```bash
export OPENAI_API_KEY='your_api_key_here'
```

## Citation
If you find our work useful, please cite:
```
@article{jin2026sonoworld,
    title={SonoWorld: From One Image to a 3D Audio-Visual Scene},
    author={Jin, Derong and Chen, Xiyi and Lin, Ming C. and Gao, Ruohan},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```

## Licence
This project is released under the MIT Licence. See [LICENSE](LICENSE).

Third-party code included in this repository may be subject to its own license
terms, as noted in the corresponding source files.
