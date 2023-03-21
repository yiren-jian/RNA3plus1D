# RNA3+1D: Sequence Convolution for 3D-RNA Modeling
This repo covers the implementation of "3+1D CNN model for RNA". The following files are tested

- [x] `C3D_model.py`
- [x] `C3D_model_new.py`
- [x] `C4D_model.py`
- [x] `Conv4d.py`
- [x] `Attn_model.py`
- [ ] `eval.py`
- [ ] `pred.py`
- [ ] `trainer.py`
- [ ] `trainer_C3D.py`

## Requirements
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.7, PyTorch 1.8.0, tensorboardX, and CUDA 11.0. You will need at least 48GB VRAM (e.g. Nvidia RTX-A6000) for running full experiments in this repo.

## Data Preparation
Prepare the data under ```data/train``` and ```data/val```. For example: ```i_0.npy``` is the `i`th positive RNA, and  ```i_j.npy``` is the `j`th negative RNA corresponding to the `i`th positive RNA. Models will be save in ```checkpoints```.
```
main
│   README.md
│   trainer.py   
│   ...
│   ...   
│
└───data   
│   └───train
│       │   0_0.npy
│       │   0_1.npy
│       │   0_2.npy
│       │   ...
│       │   ...
│       │   0_500.npy
│   
└───checkpoints
    │   model_2000.pth
    │   model_4000.pth
    │   ...
```


## Training
Assuming that you have a total of 200 (depending on your dataset) postive RNAs, and you have 500 negative samples for each RNA.
```
python trainer.py --num_RNA 200 --num_negative 500 --learning_rate 0.001 --batch_size 1 --total_steps 50000 --milestone1 30000 --milestone2 40000
```
By default, the model does not use BatchNorm. Add ```--batch_norm``` if you want to use BatchNorm in your model.

Training takes ~3.4 seconds per iteration and 6.6 GB for ```---bacth_size 1``` on a RTX-A6000.

## Visualization
The training loss/curve will be store under ```tensorboard_logdir```. Check [this](https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server) for visualizing training loss/curve on your local machine.

## Update
[2023.3.15] We add `Attn_model.py` (based on [ViT](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)) as a fully attentive model for RNA. It requires installation of `einops`
```
pip install einops
```

`Attn_model.py` has the same input/output format as `C3D_model.py` and `C4D_model.py`.

```python
### Old Convolution Model
# from C3D_model import *
# model = C3D(num_classes=2, batch_norm=opt.batch_norm).cuda()

### New Attention Model
from Attn_model import *
model = Attn_model(
    sequence_length=128,
    cubic_size=32,
    num_classes = 2,
    dim = 256,
    depth = 6,
    heads = 16,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    ).cuda()   ### use GPU
```

## Contacts
For any questions, please contact authors.

## Acknowledgment
Thanks the author for the original [Conv4d](https://github.com/ZhengyuLiang24/Conv4d-PyTorch) and [vit-pytorch](https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py) implementation.
