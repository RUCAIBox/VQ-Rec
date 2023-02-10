# VQ-Rec

PyTorch implementation of the [paper](https://arxiv.org/abs/2210.12316)
> Yupeng Hou, Zhankui He, Julian McAuley, Wayne Xin Zhao. Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders. TheWebConf 2023.

## Overview

Recently, the generality of natural language text has been leveraged to develop transferable recommender systems. The basic idea is to employ pre-trained language model~(PLM) to encode item text into item representations. Despite the promising transferability, the binding between item text and item representations might be "*too tight*", leading to potential problems such as over-emphasizing the effect of text features and exaggerating the negative impact of domain gap. To address this issue, this paper proposes **VQ-Rec**, a novel approach to learning <ins>V</ins>ector-<ins>Q</ins>uantized item representations for transferable sequential <ins>Rec</ins>ommender. The major novelty of our approach lies in the new item representation scheme: it first maps item text into a vector of discrete indices (called *item code*), and then employs these indices to lookup the code embedding table for deriving item representations. Such a scheme can be denoted as "*text* ==> *code* ==> *representation*". Based on this representation scheme, we further propose an enhanced contrastive pre-training approach, using semi-synthetic and mixed-domain code representations as hard negatives. Furthermore, we design a new cross-domain fine-tuning method based on a differentiable permutation-based network.

![](asset/model.png)

## Requirements

```
recbole==1.0.1
faiss-gpu==1.7.2
python==3.8.13
cudatoolkit==11.3.1
pytorch==1.11.0
```

## Dataset

We use the processed datasets from [UniSRec](https://github.com/RUCAIBox/UniSRec#download-datasets-and-pre-trained-model). Please **merge** (but **not** replace!!!) the current `dataset/` folder and the downloaded folders from UniSRec.

## Pre-trained Model and Item Codes

The pre-trained model is located at `pretrained/VQRec-FHCKM-300.pth`.

The pre-trained item codes (both on pre-training and downstreaem datasets) are located at `dataset/`.

## Reproduction

To quickly reproduce the reported results, you can run the following scripts.

```
python finetune.py -d Scientific -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.003
python finetune.py -d Pantry -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.003
python finetune.py -d Instruments -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.001
python finetune.py -d Arts -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.003
python finetune.py -d Office -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.003
python finetune.py -d OR -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --learning_rate=0.003
```

## Quick Start

### Data Preparation

Preparing item codes for pre-training:

```bash
python multi_pq.py --gpu_id 0
```

Preparing item codes for fine-tuning:

```bash
python pq.py --dataset Scientific --gpu_id 0
```

### Downstream

Train recommender from scratch (w/o pre-training):

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py -d Scientific --gpu_id=0
```

Fine-tune pre-trained recommender:

```bash
CUDA_VISIBLE_DEVICES=0 python finetune.py -d Scientific -p pretrained/VQRec-FHCKM-300.pth -f fix_enc --gpu_id=0
```

### Pre-train

Pre-train on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain.py --train_batch_size=2048 --gpu_id=0
```

Pre-train on multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ddp_pretrain.py
```

## Acknowledgement

The implementation is based on [UniSRec](https://github.com/RUCAIBox/UniSRec) and the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following papers as references if you use our implementations or the processed datasets.

```bibtex
@inproceedings{hou2023vqrec,
  author = {Yupeng Hou and Zhankui He and Julian McAuley and Wayne Xin Zhao},
  title = {Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders},
  booktitle={{TheWebConf}},
  year = {2023}
}

@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```

For the implementations of item code learning, thanks the amazing library [faiss](https://github.com/facebookresearch/faiss), thanks [Jingtao](https://jingtaozhan.github.io/) for the great implementation of [JPQ](https://github.com/jingtaozhan/JPQ).
