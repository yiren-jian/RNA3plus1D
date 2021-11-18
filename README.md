# RNA3+1D: Sequence Convolution for 3D-RNA modeling
This repo covers the implementation of "4D CNN model for RNA".

## Requirements
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.8.0, tensorboardX, and CUDA 11.0. You will need at least 48GB VRAM (i.e. two Nvidia RTX-A6000) for running full experiments in this repo.

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

## Contacts
For any questions, please contact authors.

## Acknowledgment
Thanks the author for the original [Conv4d](https://github.com/ZhengyuLiang24/Conv4d-PyTorch) implementation.
