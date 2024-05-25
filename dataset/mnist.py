from __future__ import print_function # 这个是用来兼容python2和python3的，因为print在python2和python3中的用法不同
import torch.utils.data as data # 这个是用来加载数据集的
from PIL import Image # 这个是用来处理图片的
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from dataset.utils import noisify


class MNIST(data.Dataset): # MNIST，这个类是用来加载MNIST数据集的
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # urls = [
    #     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', # MNIST数据集的下载地址
    #     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    # ]

    urls = [
        'https://github.com/mkolod/MNIST/raw/master/train-images-idx3-ubyte.gz',
        'https://github.com/mkolod/MNIST/raw/master/train-labels-idx1-ubyte.gz',
        'https://github.com/mkolod/MNIST/raw/master/t10k-images-idx3-ubyte.gz',
        'https://github.com/mkolod/MNIST/raw/master/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw' # 原始文件夹
    processed_folder = 'processed' # 处理后的文件夹
    training_file = 'training.pt' # 训练文件
    test_file = 'test.pt' # 测试文件

    def __init__(self, root, train=0, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, random_state=0, redux=None, full_test=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform # 标签转换，将标签转换为其他形式，比如one-hot编码，或者其他形式
        self.train = train  # training set or test set
        self.dataset='mnist' # dataset name
        self.noise_type=noise_type

        if download: # 下载数据集
            self.download()  # 下载数据集

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train == 0: # training set，训练集
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)) # process_folder是处理后的文件夹,training_file是训练文件
        elif self.train == 1: # test set，测试集
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            if not full_test:
                self.data = self.data[:7000]
                self.labels = self.labels[:7000]
        elif self.train == 2:  # validation set，验证集
            self.data, self.labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.data = self.data[:7000]
            self.labels = self.labels[:7000]

        if redux: # reduce the size of dataset，减少数据集的大小
            assert len(self.data) >= redux
            self.data = self.data[:redux]
            self.labels = self.labels[:redux]

        if noise_type == 'clean': # clean dataset，干净数据集
            self.labels = np.asarray(self.labels) # 将标签转换为numpy数组
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.int32)
        else: # noisy dataset，嘈杂数据集，即有噪声的数据集
            self.labels = np.asarray(self.labels) # 将标签转换为numpy数组
            self.noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, # 生成噪声
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type, noise_rate=noise_rate,
                                                                random_state=random_state)
            self.noisy_labels = self.noisy_labels.squeeze() # 去掉维度为1的维度
            self.noise_or_not = self.noisy_labels == self.labels  # 生成噪声标签


    def __getitem__(self, index): # 获取数据集中的数据
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.noise_type != 'clean': # noisy dataset，嘈杂数据集
            img, target = self.data[index], self.noisy_labels[index] # 有噪声的标签
        else: # clean dataset，干净数据集
            img, target = self.data[index], self.labels[index] # 干净的标签

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self): # 检查数据集是否存在
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url) # 下载数据集
            data = urllib.request.urlopen(url)  # urllib.request.urlopen()方法用于打开一个URL地址
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
