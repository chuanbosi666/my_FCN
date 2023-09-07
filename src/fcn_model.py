from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    模块包装器，从模型返回中间层
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    此外，它只能查询直接分配给模型的子模块。
    分配给模型的子模块。因此，如果传入 `model`,`model.feature1` 可以被返回，
    但不能返回 `model.feature1.layer2`。但不能返回 `model.feature1.layer2`。
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
              # 判断return_layers是否是以一个集合，并且是否在model的模块中
              # .issubset是用于检查一个集合是否是另一个集合的子集的方法
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers  # 因为这里的return）layers是一个字典的格式，所以是为了根据传入的内容来判断是使用分类器还是辅助分类器
        return_layers = {str(k): str(v) for k, v in return_layers.items()}  # 强制性的将返回层的key和value转换成字符型

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()  # 创建一个有序字典
        for name, module in model.named_children(): # 这里的model 是我们的backbone
             # 这里的return_layer会根据使用的骨干网络的不同而不同，例如resnet50的为layer4为输出，以字典的形式保存
            layers[name] = module  # 将我们的backbone放入layers中去
            if name in return_layers:  # 如果name在return_layers中，就将其放入layers中去
                del return_layers[name]  # 不在的话，删除return_layers中的name
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  #将构建好的backbone放入父类中去
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()  #这是模型的正向传播
        for name, module in self.items():
            x = module(x)  #将图像放进模型中去
            if name in self.return_layers:  #这个判断的作用是来使用out还是辅助分类器的out
                out_name = self.return_layers[name]  #当为layer3时，会使用辅助分类器，为layer4时会使用正常的分类器
                out[out_name] = x  # 把我们得到的预测放入分类器中去
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
            主干应该返回一个 OrderedDict[Tensor]，其中 "out "表示最后使用的特征图，
            "aux "表示使用了辅助分类器。
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):  #这里有backbone和分类器，辅助分类器默认不使用
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:  #输入的是张量 [480,480,3]
        input_shape = x.shape[-2:]  # 这里的input_shape是[480，480]
        # contract: features is a dict of tensors
        features = self.backbone(x) #经过backbone会得到特征图

        result = OrderedDict()  # 创建一个有序字典
        x = features["out"]  # 从下面可知，这里的x是我们的layer4得到的特征图
        x = self.classifier(x)  # 放入FCNHead中
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)  #使用双线性插值，
        # 对应的是传进来的参数，输出的size为480，方式是双线性插值，最后一个参数是是否将输入的四个点与输出的四个点对齐
        # 经过大致的运行发现，双线性插值和膨胀卷积效果差不多，为此我们选择了比较高效的双线性插值
        result["out"] = x  #一个简单的分类器使用，参数传递

        if self.aux_classifier is not None:  # 对辅助分类器的使用，不过一般不使用
            x = features["aux"] # 这里的x是我们的layer3得到的特征图
            x = self.aux_classifier(x) # 放入FCNHead中
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False) #使用双线性插值
            result["aux"] = x #一个简单的分类器使用，参数传递

        return result


class FCNHead(nn.Sequential):  # 一个分类头
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4  # 这里的in_channels是我们的layer4的输出通道数，也就是2048，这里的channels是我们的类别数，也就是21
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 对卷积进行的参数进行设定，卷积核大小为3，填充为1，不加偏置
            nn.BatchNorm2d(inter_channels),  # 对输入通道进行BN层的处理
            nn.ReLU(),  # 使用RELU激活
            nn.Dropout(0.1),    # 使用dropout
            nn.Conv2d(inter_channels, channels, 1)  # 对卷积进行的参数进行设定，卷积核大小为1，不加偏置
        ]

        super(FCNHead, self).__init__(*layers) #将上面的层放入父类中去


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])  # 这个是我们的骨干网络，使用resnet50，对replace_stride_with_dilation进行了设定

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    out_inplanes = 2048 # 这里的输出通道数为2048
    aux_inplanes = 1024  # 这里的辅助分类器的输出通道数为1024

    return_layers = {'layer4': 'out'}  # 对返回层进行了设定
    if aux:
        return_layers['layer3'] = 'aux'  # 对辅助分类器的返回层进行了设定，这个就是对layer3的输出进行了设定
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)  # 将backbone和返回层放入IntermediateLayerGetter中去

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)  # 对辅助分类器进行了设定

    classifier = FCNHead(out_inplanes, num_classes)  # 对分类器进行了设定

    model = FCN(backbone, classifier, aux_classifier)  # 将backbone，分类器，辅助分类器放入FCN中去

    return model  # 返回模型


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):  # 这个是使用resnet101的模型，大致设定和上面的差不多，会有些许区别
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])  # 设定骨干网络

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024  # 这里分别设定输出和辅助分类器的输出大小

    return_layers = {'layer4': 'out'}  # 对返回层进行了设定
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
