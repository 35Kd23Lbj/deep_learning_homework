import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from dataset.utils import noisify
from torchvision import transforms
import torch

class MedicalImageDataset(Dataset): # 定义一个数据集类
    def __init__(self, root_dir, train=0, transform=None, target_transform=None,
                 noise_type=None, noise_rate=0.2, random_state=0):
        # 参数：数据集的根目录，训练集或测试集，transform，target_transform，噪声类型，噪声率，随机种子
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将所有图像调整为相同的大小
            transform
        ]) if transform else transforms.Resize((224, 224))  # 如果transform不为空，则将所有图像调整为相同的大小，否则将所有图像调整为(224, 224)的大小
        self.target_transform = target_transform
        self.image_files = []
        self.labels = []
        self.class_names = os.listdir(root_dir)  # 获取所有类别的名称
        self.train = train  # training set or test set
        self.dataset = 'medical'  # dataset name
        self.noise_type = noise_type # 'clean' or 'instance'

        # 遍历每个类别的目录，获取所有图像文件的路径和对应的标签
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                _, ext = os.path.splitext(image_name)
                if ext.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:  # 只有当文件是图像文件时才添加
                    self.image_files.append(os.path.join(class_dir, image_name))
                    self.labels.append(label)

        # 打乱并划分数据集
        indices = np.arange(len(self.image_files))
        np.random.shuffle(indices)
        train_split = int(len(indices) * 0.8)
        val_split = int(len(indices) * 0.1)
        if train == 0: # training set，训练集
            self.image_files = [self.image_files[i] for i in indices[:train_split]]
            self.labels = [self.labels[i] for i in indices[:train_split]]
        elif train == 1: # validation set，验证集
            self.image_files = [self.image_files[i] for i in indices[train_split:train_split + val_split]]
            self.labels = [self.labels[i] for i in indices[train_split:train_split + val_split]]
        else:  # test set，测试集
            self.image_files = [self.image_files[i] for i in indices[train_split + val_split:]]
            self.labels = [self.labels[i] for i in indices[train_split + val_split:]]


        if noise_type == 'clean':  # clean dataset，干净数据集
            self.labels = np.asarray(self.labels)  # 将标签转换为numpy数组
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.int32)
        else:  # noisy dataset，嘈杂数据集，即有噪声的数据集
            self.labels = np.asarray(self.labels)  # 将标签转换为numpy数组
            self.noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset,  # 生成噪声
                                                                nb_classes=len(self.class_names),
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type, noise_rate=noise_rate,
                                                                random_state=random_state)
            self.noisy_labels = self.noisy_labels.squeeze()  # 去掉维度为1的维度
            self.noise_or_not = self.noisy_labels == self.labels  # 生成噪声标签

    def __getitem__(self, index):  # 获取数据集中的数据
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')  # convert image to RGB
        if self.transform:
            img = self.transform(img)
        if self.noise_type != 'clean':  # noisy dataset，嘈杂数据集
            target = self.noisy_labels[index]  # 有噪声的标签
        else:  # clean dataset，干净数据集
            target = self.labels[index]  # 干净的标签
        if self.target_transform: # 对标签进行转换
            target = self.target_transform(target) # 对标签进行转换
        return img, target, index

    def __len__(self):
        return len(self.image_files)  # 返回数据集的大小



