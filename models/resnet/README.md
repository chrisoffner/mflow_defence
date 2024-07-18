# PyTorch ResNet-50 CIFAR-10 classifier

Pre-trained checkpoint `resnet50_cifar10.pt` achieves a train accuracy of 99.46% and test accuracy of **96.27%**.

To train a ResNet-50 from scratch (will start from [pre-trained ImageNet weights](https://pytorch.org/vision/stable/models.html#initializing-pre-trained-models)), run: 

```sh
python train_resnet.py
```

To resume training from a checkpoint, run:

```sh
python train_resnet.py --checkpoint <path-to-checkpoint> [--checkpoint_variables <path-to-checkpoint-variables>] 
```

Acknowledgments: implementation borrows heavily from [Fine-Tuning-ResNet50-Pretrained-on-ImageNet-for-CIFAR-10](https://github.com/sidthoviti/Fine-Tuning-ResNet50-Pretrained-on-ImageNet-for-CIFAR-10) and [PyTorch's Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
 

