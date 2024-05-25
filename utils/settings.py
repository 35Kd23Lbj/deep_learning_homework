import sys
import argparse

def get_args(): # 获取参数
    if sys.platform == 'darwin':  # 如果是macOS系统
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/ISIC-Archive-Downloader/Data_sample_balanced'
        mnist_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/MNIST'
        cifar10_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar10'
        cifar100_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar100'
        pcam_root = "/Users/jiarunliu/Documents/BUCT/Label_517/dataset/PatchCamelyon"
        batch_size = 8
        device = 'cpu'
        data_device = 0
        noise_type = 'sn' # 默认噪声类型为sn，sn是指Symmetric Noise，即对称噪声，这种噪声是加入噪声标签的一种方式，即对标签进行随机翻转，使得标签错误。
        stage1 = 1 # 默认stage1为1
        stage2 = 3 # 默认stage2为3
    elif sys.platform == 'linux': # 如果是linux系统
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/home/fgldlb/Documents/ISIC-Archive-Downloader/NewData'
        pcam_root = "/home/fgldlb/Documents/dataset/PatchCamelyon"
        mnist_root = './data/mnist'
        cifar10_root = './data/cifar10'
        cifar100_root = './data/cifar100'
        batch_size = 32
        device = 'cuda:0'
        data_device = 1
        noise_type = 'sn' # 默认噪声类型为sn，sn是指Symmetric Noise，即对称噪声
        stage1 = 70
        stage2 = 200
    else: # 如果是其他系统，如windows系统
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = None
        mnist_root = './data/mnist' # mnist数据集路径
        medical_root = "./data/medical" # medical数据集路径
        cifar10_root = '/data/cifar10'
        cifar100_root = '/data/cifar100'
        pcam_root = None
        batch_size = 16
        # device = 'cpu' # 默认设备为cpu
        device = 'cuda:0' # 默认设备为cuda:0，即GPU
        data_device = 0
        # noise_type = 'clean' # 默认噪声类型为clean，clean是指无噪声
        noise_type = 'sn' # 默认噪声类型为sn，sn是指Symmetric Noise，即对称噪声
        stage1 = 10 # 默认stage1为10，stage1是第一阶段，即第一阶段的训练轮数，第一阶段是在更新标签前的训练轮数，用来训练两个流，即两个网络，然后用两个网络的预测结果来更新标签
        stage2 = 20 # 默认stage2为20，stage2是第二阶段，即第二阶段的训练轮数，第二阶段是在更新标签后的训练轮数，用来训练模型，即用更新后的标签来训练模型

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training') # 创建一个参数解析器
    # normal parameters 参数设置
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int, # 默认batch_size为xx，这里的batch_size是指每个批次的样本数
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, # 默认学习率为0.1
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-5, type=float, # 默认学习率为0.1
                        metavar='H-P', help='initial learning rate of stage3')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', # 默认动量为0.9
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, # 默认权重衰减为1e-4
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--backbone', dest="backbone", default="resnet50", type=str,
                        help="backbone for PENCIL training")
    parser.add_argument('--optim', dest="optim", default="SGD", type=str, # 默认优化器为SGD
                        choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adadelta', 'Adagrad', 'mix'],
                        help="Optimizer for PENCIL training")
    parser.add_argument('--scheduler', dest='scheduler', default=None, type=str, choices=['cyclic', None, "SWA"],
                        help="Optimizer for PENCIL training")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', # 默认数据加载工作线程数为4
                        help='number of data loading workers (default: 4)')
    # Co-teaching parameters
    parser.add_argument('--forget-rate', '--fr', '--forget_rate', default=0.2, type=float, # 默认forget_rate为0.2，forget_rate是遗忘率
                        metavar='H-P', help='Forget rate. Suggest same with noisy density.')
    parser.add_argument('--num-gradual', '--ng', '--num_gradual', default=10, type=int, # 默认num_gradual为10，num_gradual是渐进数量
                        metavar='H-P', help='how many epochs for linear drop rate, can be 5, 10, 15. '
                                            'This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', default=1, type=float, # 默认exponent为1，exponent是指数
                        metavar='H-P', help='exponent of the forget rate, can be 0.5, 1, 2. '
                                            'This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--loss-type', dest="loss_type", default="coteaching_plus", type=str,
                        choices=['coteaching_plus', 'coteaching'],
                        help="loss type: [coteaching_plus, coteaching]")
    parser.add_argument('--warmup', '--wm', '--warm-up', default=0, type=float,
                        metavar='H-P', help='Warm up process eopch, default 0.')
    parser.add_argument('--linear-num', '--linear_num', default=256, type=int,
                        metavar='H-P', help='how many epochs for linear drop rate, can be 5, 10, 15. '
                                            'This parameter is equal to Tk for R(T) in Co-teaching paper.')
    # PENCIL parameters PENCIL参数，PENCIL是一种半监督学习方法，用于解决噪声标签问题
    # PENCIL的核心是两个流，一个流用于预测，一个流用于选择，选择的标准是预测结果的不一致性
    # 即两个流的预测结果不一致的样本会被选择出来，然后用于更新标签，这样可以减少噪声标签的影响，提高模型的泛化能力
    # PENCIL的两个流可以是同一个网络，也可以是不同的网络，这里我们使用两个不同的网络，通过损失函数来更新标签，然后再用更新后的标签来训练模型
    parser.add_argument('--alpha', default=0.4, type=float, # 默认alpha为0.4，alpha是指Compatibility Loss的系数
                        metavar='H-P', help='the coefficient of Compatibility Loss')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='the coefficient of Entropy Loss')
    parser.add_argument('--lambda1', default=200, type=int, # 默认lambda1为200，lambda1是指Compatibility Loss的系数
                        metavar='H-P', help='the value of lambda, ')
    parser.add_argument('--K', default=10.0, type=float, )
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # 默认训练轮数为xx，这里的训练轮数是指训练的总轮数，包括两个阶段，第一阶段为stage1，第二阶段为stage2，stage1和stage2的值可以自己设置
    parser.add_argument('--epochs', default=50, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--stage1', default=stage1, type=int, # 默认stage1为1，stage1是第一阶段
                        metavar='H-P', help='number of epochs utill stage1')
    parser.add_argument('--stage2', default=stage2, type=int, # 默认stage2为3，stage2是第二阶段
                        metavar='H-P', help='number of epochs utill stage2')
    # Nosie settings 噪声设置
    parser.add_argument('--noise', default=0.20, type=float,  # 默认噪声密度为0.20，噪声密度是指噪声标签的比例
                        help='noise density of data label')
    parser.add_argument('--noise_type', default='sn',  choices=['clean', 'sn', 'pairflip'],type=str, # 默认噪声类型为sn，噪声类型是指噪声标签的类型
                        help='noise tyoe of data label')
    # Data settings 数据设置
    parser.add_argument("--dataset", dest="dataset", default='mnist', type=str, # 默认数据集为,这里可以自定义数据集
                        choices=['mnist', 'cifar10', 'cifar100', 'cifar2', 'isic', 'clothing1m', 'pcam','medical'],
                        # 数据集有mnist, cifar10, cifar100, cifar2, isic, clothing1m, pcam, medical
                        help="model input image size")
    parser.add_argument("--image_size", dest="image_size", default=224, type=int, # 默认图像大小为224
                        help="model input image size")
    parser.add_argument('--classnum', default=3, type=int, # 默认类别数为3，类别数是指数据集的类别数，如mnist为10
                        metavar='H-P', help='number of train dataset classes')
    parser.add_argument('--device', dest='device', default=device, type=str, # 默认设备为cuda:0，即GPU
                        help='select gpu')
    parser.add_argument('--data_device', dest="data_device", default=data_device, type=int,
                        help="Dataset loading device, 0 for hardware 1 for RAM. Default choice is 1. "
                             "Please ensure your computer have enough capacity!")
    parser.add_argument('--dataRoot',dest='root',default=medical_root, # 默认数据集路径为medical_root,即./data/medical
                        type=str,metavar='PATH',help='where is the dataset')
    parser.add_argument('--datanum', default=15000, type=int,
                        metavar='H-P', help='number of train dataset samples')
    parser.add_argument('--train-redux', dest="train_redux", default=None, type=int,
                        help='train data number, default None')
    parser.add_argument('--test-redux', dest="test_redux", default=None, type=int,
                        help='test data number, default None')
    parser.add_argument('--val-redux', dest="val_redux", default=None, type=int,
                        help='validate data number, default None')
    parser.add_argument('--full-test', dest="full_test", default=False, type=bool,
                        help='use full test set data, default False')
    parser.add_argument('--random-ind-redux', dest="random_ind_redux", default=False, type=bool,
                        help='use full test set data, default False')
    # Curriculum settings 课程设置
    parser.add_argument("--curriculum", dest="curriculum", default=1, type=int,
                        help="curriculum in label updating")
    parser.add_argument("--cluster-mode", dest="cluster_mode", default='dual', type=str, choices=['dual', 'single', 'dual_PCA'],
                        help="curriculum in label updating")  # 设置聚类模式，有三种选择，分别为dual, single, dual_PCA，分别对应两个流，单流，两个流PCA
    parser.add_argument("--dim-reduce", dest="dim_reduce", default=256, type=int,
                        help="Curriculum features dim reduce by PCA")
    parser.add_argument("--mix-grad", dest="mix_grad", default=1, type=int,
                        help="mix gradient of two-stream arch, 1=True")
    parser.add_argument("--discard", dest="discard", default=0, type=int,
                        help="only update discard sample's label, 1=True")
    parser.add_argument("--gamma", dest="gamma", default=0.6, type=int,
                        help="forget rate schelduler param")
    parser.add_argument("--finetune-schedule", '-fs', dest="finetune_schedule", default=0, type=int,
                        help="forget rate schelduler param")
    # trainer settings
    parser.add_argument('--dir', dest='dir', default="experiment/test-debug", type=str, # 默认保存路径为experiment/test-debug
                        metavar='PATH', help='save dir')
    parser.add_argument('--random-seed', dest='random_seed', default=0, type=int, # 默认随机种子为0，如果设置了随机种子，那么每次训练的结果都是一样的
                        metavar='N', help='pytorch random seed, default 0.')
    args = parser.parse_args()

    # Setting for different dataset 数据集设置
    if args.dataset == "isic": # 如果数据集为isic
        print("Training on ISIC") # 输出训练数据集为ISIC
        args.backbone = 'resnet50' # 默认backbone为resnet50
        args.image_size = 224 # 默认图像大小为224
        args.classnum = 2
        args.input_dim = 3 # 输入维度为3
    elif args.dataset == 'mnist': # 如果数据集为mnist
        print("Training on mnist")
        args.backbone = 'cnn' # 默认backbone为cnn
        if args.root == isic_root:
            args.root = mnist_root
        args.batch_size = 128
        args.image_size = 28
        args.classnum = 10
        args.input_dim = 1
        args.linear_num = 144
        args.datanum = 60000
        args.lr = 0.001
        args.lr2 = 0.0001
    elif args.dataset == 'pcam': # 如果数据集为pcam
        if args.root == isic_root:
            args.root = pcam_root
        args.backbone = 'densenet169'
        args.batch_size = 128
        args.image_size = 96
        args.dim_reduce = 128
        args.classnum = 2
        args.input_dim = 3
        args.stage1 = 70
        args.stage2 = 200
        args.epochs = 320
        args.datanum = 262144
        args.train_redux = 26214
        args.test_redux = 3276
        args.val_redux = 3276
        args.random_ind_redux = False
    elif args.dataset == 'medical': # 如果数据集为medical
        print("Training on medical")
        args.backbone = 'efficientnet_b0' # 默认backbone为efficientnet_b0
        args.image_size = 224
        args.classnum = 3 # 类别数为3
        args.input_dim = 3 # 输入维度为3,即RGB图像
        args.datanum = 15000
        args.lr = 0.001
        args.lr2 = 0.0001
        args.batch_size = 16
        args.epochs = 50 # 训练轮数为50
        args.stage1 = 10 # 第一阶段为10
        args.stage2 = 20 # 第二阶段为20
        args.root = medical_root
    else:
        print("Use default setting")

    return args