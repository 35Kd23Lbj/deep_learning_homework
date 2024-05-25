import os
import copy
import json
import datetime
import numpy as np
from os.path import join

import torch

import torchvision

# from dataset.cifar import CIFAR10, CIFAR100  # 从dataset文件夹中导入CIFAR10和CIFAR100
from dataset.mnist import MNIST
from dataset.ISIC import ISIC
# from dataset.clothing1m import Clothing1M  # 从dataset文件夹中导入Clothing1M
from dataset.PatchCamelyon import PatchCamelyon
from dataset.medical import MedicalImageDataset # 从dataset文件夹中导入MedicalImageDataset

from models.densenet import densenet121, densenet161, densenet169, densenet201
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from models.coteaching_model import MLPNet, CNN_small, CNN
from models.efficientnet import efficientnet_b0 # 从models文件夹中导入efficientnet_b0

class BasicTrainer(object): # BasicTrainer是一个基础的训练器，用于训练模型

    def __init__(self, args):  # 初始化函数
        self._get_args(args) # 获取参数
        if self.args.random_seed is not None:
            torch.manual_seed(self.args.random_seed)

    def _save_meta(self): # 保存元数据，包括参数设置
        # save meta data
        print(vars(self.args))
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        with open(join(self.args.dir, "settings-{}.json".format(nowTime)), 'w') as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)


    def _get_args(self, args): # 获取参数
        self.args = args

        # addition args 添加参数
        self.args.checkpoint_dir = join(self.args.dir, "checkpoint.pth.tar") # 检查点目录
        self.args.modelbest_dir = join(self.args.dir, "model_best.pth.tar") # 最佳模型目录
        self.args.record_dir = join(self.args.dir, 'record.json') # 记录目录
        self.args.y_file = join(self.args.dir, "y.npy") # y文件，用于记录标签，numpy格式，后缀名为.npy
        self.best_prec1 = 0

    def _get_model(self, backbone): # 获取模型, backbone是指定的模型,如resnet50, resnet101等
        if backbone == 'resnet18':
            model = resnet18(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'resnet34':
            model = resnet34(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'resnet50':
            model = resnet50(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'resnet101':
            model = resnet101(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'resnet152':
            model = resnet152(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'preact_resnet18':
            model = PreActResNet18(num_classes=self.args.classnum, input_size=self.args.image_size,
                                   input_dim=self.args.input_dim).to(self.args.device)
        elif backbone == 'preact_resnet34':
            model = PreActResNet34(num_classes=self.args.classnum, input_size=self.args.image_size,
                                   input_dim=self.args.input_dim).to(self.args.device)
        elif backbone == 'preact_resnet50':
            model = PreActResNet50(num_classes=self.args.classnum, input_size=self.args.image_size,
                                   input_dim=self.args.input_dim).to(self.args.device)
        elif backbone == 'preact_resnet101':
            model = PreActResNet101(num_classes=self.args.classnum, input_size=self.args.image_size,
                                    input_dim=self.args.input_dim).to(self.args.device)
        elif backbone == 'preact_resnet152':
            model = PreActResNet152(num_classes=self.args.classnum, input_size=self.args.image_size,
                                    input_dim=self.args.input_dim).to(self.args.device)
        elif backbone == 'densenet121':
            model = densenet121(num_classes=self.args.classnum, pretrained=True).to(self.args.device)
        elif backbone == 'densenet161':
            model = densenet161(num_classes=self.args.classnum, pretrained=True).to(self.args.device)
        elif backbone == 'densenet169':
            model = densenet169(num_classes=self.args.classnum, pretrained=True).to(self.args.device)
        elif backbone == 'densenet201':
            model = densenet201(num_classes=self.args.classnum, pretrained=True).to(self.args.device)
        elif backbone == 'mlp':
            model = MLPNet().to(self.args.device)
        elif backbone == 'cnn_small' or backbone == "CNN_SMALL":
            model = CNN_small(self.args.classnum).to(self.args.device)
        elif backbone == "cnn" or backbone == "CNN":
            model = CNN(n_outputs=self.args.classnum, input_channel=self.args.input_dim, linear_num=self.args.linear_num).to(self.args.device)
        elif backbone == "efficientnet_b0":
            model = efficientnet_b0(num_classes=self.args.classnum).to(self.args.device)
        else:
            print("No matched backbone. Using ResNet50...")
            model = resnet50(pretrained=True, num_classes=self.args.classnum,
                             input_size=self.args.image_size).to(self.args.device)

        return model

    def _get_optim(self, parm, optim="SGD", scheduler=None, lr=None): # 获取优化器
        if optim == "SGD" or optim == "sgd":
            optimizer = torch.optim.SGD(parm, lr=lr if lr else self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif optim == "adam" or optim == "Adam" or optim == "ADAM":
            optimizer = torch.optim.Adam(parm, lr=lr if lr else self.args.lr)
        elif optim == "adamw" or optim == "AdamW":
            optimizer = torch.optim.AdamW(parm, lr=lr if lr else self.args.lr)
        elif optim == "RMSprop" or optim == "rmsprop":
            optimizer = torch.optim.RMSprop(parm, lr=lr if lr else self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif optim == "Adadelta":
            optimizer = torch.optim.Adadelta(parm, lr=lr if lr else self.args.lr)
        elif optim == "Adagrad":
            optimizer = torch.optim.Adagrad(parm, lr=lr if lr else self.args.lr)
        else:
            NotImplementedError("No Such Optimizer Implemented: {}".format(optim))

        return optimizer

    def _get_dataset_isic(self): # 获取数据集
        transform = torchvision.transforms.Compose([ # 使用torchvision.transforms.Compose()函数将多个数据预处理操作组合在一起
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(degrees=[-180, 180]),
            torchvision.transforms.Resize(self.args.image_size),
            torchvision.transforms.ToTensor(),
        ])
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.args.image_size),
            torchvision.transforms.ToTensor(),
        ])

        trainset = ISIC(root=self.args.root,
                                         train=0,
                                         transform=transform,
                                         noise_type=self.args.noise_type,
                                         noise_rate=self.args.noise,
                                         device=self.args.data_device,
                                         redux=self.args.train_redux,
                                         image_size=self.args.image_size)
        testset = ISIC(root=self.args.root,
                                        train=1,
                                        transform=transform1,
                                        noise_type='clean',
                                        noise_rate=self.args.noise,
                                        device=self.args.data_device,
                                        redux=self.args.test_redux,
                                        image_size=self.args.image_size)
        valset = ISIC(root=self.args.root,
                                       train=2,
                                       transform=transform1,
                                       noise_type='clean',
                                       noise_rate=self.args.noise,
                                       device=self.args.data_device,
                                       redux=self.args.val_redux,
                                       image_size=self.args.image_size)

        return trainset, testset, valset

    def _get_dataset_pcam(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(degrees=[-90, 90]),
            torchvision.transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
            torchvision.transforms.Resize(self.args.image_size),
            torchvision.transforms.ToTensor(),
        ])
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.args.image_size),
            torchvision.transforms.ToTensor(),
        ])

        trainset = PatchCamelyon(root=self.args.root,
                                 train=0,
                                 transform=transform,
                                 noise_type=self.args.noise_type,
                                 noise_rate=self.args.noise,
                                 redux=self.args.train_redux,
                                 random_ind_redux=self.args.random_ind_redux)
        testset = PatchCamelyon(root=self.args.root,
                                train=1,
                                transform=transform1,
                                noise_type='clean',
                                noise_rate=0,
                                redux=self.args.test_redux,
                                random_ind_redux = self.args.random_ind_redux)
        valset = PatchCamelyon(root=self.args.root,
                               train=2,
                               transform=transform1,
                               noise_type='clean',
                               noise_rate=0,
                               redux=self.args.val_redux,
                               random_ind_redux=self.args.random_ind_redux)

        return trainset, testset, valset

    def _get_dataset_mnist(self):
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.RandomPerspective(),
            torchvision.transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
            torchvision.transforms.ToTensor(),
        ])
        transform = torchvision.transforms.ToTensor()
        trainset = MNIST(root=self.args.root,
                         download=True,
                         train=0,
                         transform=transform1,
                         noise_type=self.args.noise_type,
                         noise_rate=self.args.noise,
                         redux=self.args.train_redux,
                         )
        testset = MNIST(root=self.args.root,
                        download=True,
                        train=1,
                        transform=transform,
                        noise_type='clean',
                        noise_rate=0,
                        redux=self.args.test_redux,
                        full_test=self.args.full_test,
                        )
        valset = MNIST(root=self.args.root,
                       download=True,
                       train=2,
                       transform=transform,
                       noise_type='clean',
                       noise_rate=0,
                       redux=self.args.val_redux,
                       )

        return trainset, testset, valset

    def _get_dataset_medical(self): # 获取医学数据集，包括训练集、测试集和验证集
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(degrees=[-90, 90]),
            torchvision.transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
            torchvision.transforms.Resize(self.args.image_size),
            torchvision.transforms.ToTensor(),
        ])
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.args.image_size), # 将所有图像调整为相同的大小
            torchvision.transforms.ToTensor(), # 将图像转换为张量
        ])
        # 训练集是噪声数据集，测试集和验证集是干净数据集
        trainset = MedicalImageDataset(root_dir=self.args.root,
                                        transform=transform,
                                        train=0,
                                        noise_type=self.args.noise_type,
                                        noise_rate=self.args.noise, # 噪声率

                                        )
        testset = MedicalImageDataset(root_dir=self.args.root,
                                        transform=transform1,
                                        train=2,
                                        noise_type='clean', # 干净数据集
                                        noise_rate=0,

                                        )
        valset = MedicalImageDataset(root_dir=self.args.root,
                                        transform=transform1,
                                        train=1,
                                        noise_type='clean', # 干净数据集
                                        noise_rate=0,

                                        )
        return trainset, testset, valset


    def _load_data(self): # 加载数据
        if self.args.dataset == 'isic':
            trainset, testset, valset = self._get_dataset_isic()
        elif self.args.dataset == 'mnist':
            trainset, testset, valset = self._get_dataset_mnist()
        elif self.args.dataset == 'pcam':
            trainset, testset, valset = self._get_dataset_pcam()
        elif self.args.dataset == 'medical':
            trainset, testset, valset = self._get_dataset_medical()
        else:
            NotImplementedError("Dataset [{}] Was Not Been Implemented".format(self.args.dataset))
        # 加载数据集
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size,
                                                       shuffle=True, num_workers=self.args.workers,
                                                       pin_memory=True if self.args.data_device == 1 else False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size,
                                                      shuffle=False, num_workers=self.args.workers,
                                                      pin_memory=True if self.args.data_device == 1 else False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.args.batch_size,
                                                     shuffle=False, num_workers=self.args.workers,
                                                     pin_memory=True if self.args.data_device == 1 else False)

        self.train_batch_num = len(trainloader)
        self.test_batch_num = len(testloader)
        self.val_batch_num = len(valloader)

        self.train_data_num = len(trainset)
        self.test_data_num = len(testset)
        self.val_data_num = len(valset)

        self.noise_or_not = trainset.noise_or_not
        self.clean_labels = trainset.labels

        print("Train num: {}\tTest num: {}\tVal num: {}".format(len(trainset), len(testset), len(valset)))
        return trainloader, testloader, valloader