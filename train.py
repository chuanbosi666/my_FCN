import os
import time
import datetime

import torch
# 导入我们的模型，使用的是resnet50
from src import fcn_resnet50
# 导入我们的训练要用的一些工具
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
# 导入我们的数据集
from my_dataset import VOCSegmentation
import transforms as T


class SegmentationPresetTrain:
    # 对训练的数据进行变换，设定一些参数，像是图像的大小，裁剪的大小，是否进行水平翻转，以及图像的均值和方差
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size) #设定最大尺寸

        trans = [T.RandomResize(min_size, max_size)]  # 对图像进行随机的缩放
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))  # 对图像进行水平翻转
        trans.extend([  # 对图像进行裁剪，转换为张量，以及对图像进行归一化
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)  # 将上面的操作放入一个列表中

    def __call__(self, img, target):
        return self.transforms(img, target)  #这个函数的作用是调用这个类之后会自动调用这个函数


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target) #这个类和上面的类似，只不过对图片的变换比较少


def get_transform(train):
    base_size = 520
    crop_size = 480  # 对图像的大小进行设定，以及裁剪的大小
# 对训练和验证的数据进行不同的变换
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)  #根据传进来的参数来运行


def create_model(aux, num_classes, pretrain=True):
    model = fcn_resnet50(aux=aux, num_classes=num_classes)  # 对模型进行创建，传入参数

    if pretrain: # 对预训练权重进行加载
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')  #看使用预训练权重，传入路径

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]  # 当我们使用voc数据集时，类别数为21，可以使用预训练权重，
                    #当使用我们自己的数据集的时候，就要将我们分类器的第四层的权重删去，也就是最后一个卷积的通道数删去，保证我们的训练正常进行
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False) #对权重进行加载，strict=False是为了防止权重shape不一致报错
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)  #第一个打印的是我们在删去classifier.4的权重，第二个是我们没使用的权重

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 对设备进行选择，如果有GPU就使用GPU，没有就使用CPU
    batch_size = args.batch_size  # 对batch_size进行设定
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1 #对参数的设定

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt") #对传入数据的路径，参数'year'的传入，对图像转换的设定为训练模式，以及图片索引的文件

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")  # 同上

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 对这几个取最小值作为同时运行的最小数值
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,  # 对数据集进行加载，传入参数，对数据集进行打乱，以及是否将数据放入GPU中
                                               collate_fn=train_dataset.collate_fn)  #  冻结BN层

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)  # 同上

    model = create_model(aux=args.aux, num_classes=num_classes)  # 传入参数
    model.to(device) #放进设备中

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ] # 对参数进行优化，这里是对backbone和classifier的参数进行优化，requires_grad是为了防止我们的参数被冻结

    if args.aux:  #下面的都是将参数传进相关的位置
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    ) # 对优化器进行设定，这里使用的是SGD，传入参数

    scaler = torch.cuda.amp.GradScaler() if args.amp else None # 对混合精度进行设定，如果使用混合精度，就使用GradScaler，否则就为None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:  # 对模型进行加载
        checkpoint = torch.load(args.resume, map_location='cpu')  # 传入路径
        model.load_state_dict(checkpoint['model'])  #   加载模型
        optimizer.load_state_dict(checkpoint['optimizer'])  #   加载优化器
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  #   加载学习率更新策略
        args.start_epoch = checkpoint['epoch'] + 1   #   加载开始的epoch
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])  #   加载混合精度

    start_time = time.time()  # 记录时间
    for epoch in range(args.start_epoch, args.epochs):  # 对epoch进行循环
        # train for one epoch, printing every 10 iterations
        #训练一个epoch，每10次迭代打印一次，迭代次数为训练集的大小除以batch_size
        # 这里的train_one_epoch是我们自己写的函数，对模型进行训练，传入参数，像是模型，优化器，训练集，设备，epoch，学习率更新策略，打印频率，混合精度
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # evaluate on the val dataset
        # 对验证集进行验证，evaluate是我们自己写的函数，对模型进行验证，传入参数，像是模型，验证集，设备，类别数
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat) # 将验证集的结果转换为字符串
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:  # 将结果写入txt文件中
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}  # 将模型，优化器，学习率更新策略，epoch，参数放入字典中
        if args.amp:
            save_file["scaler"] = scaler.state_dict()  # 将混合精度放入字典中
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))  #保存模型

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')  # 对动量进行设定，动量是为了防止我们的梯度下降过程中陷入局部最优解
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
