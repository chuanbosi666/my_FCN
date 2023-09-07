import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):  # 一个三乘三的卷积
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1): # 一个一乘一的卷积
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):  # 对于使用的Bottleneck进行定义
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4  # 先设定我们的膨胀率为4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None): #参数定义
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 设置图片的宽度，保证能够正常的处理  这里的width为planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  #设定传进来的输入维度和输出
        self.bn1 = norm_layer(width)  #设定传入参数  (width, out_planes, kernel_size=1,stride=1,bias=False)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  #设定传进去的3x3卷积的参数
        self.bn2 = norm_layer(width)#同上
        self.conv3 = conv1x1(width, planes * self.expansion)  # 传进来的参数输出时会进行空洞卷积
        self.bn3 = norm_layer(planes * self.expansion)  #  同上
        self.relu = nn.ReLU(inplace=True)  # 会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的值传递
        self.downsample = downsample  #这里的下采样就是一个步幅为一的1x1卷积
        self.stride = stride

    def forward(self, x):
        identity = x  

        out = self.conv1(x) #经过1x1卷积，BN层和RELU激活
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 经过3x3卷积，BN层，RELU激活
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  #经过1x1卷积，BN层，当为Bottleneck1时会进行下采样，然后再相加RELU激活，而当为Bottleneck2时，会直接相加再激活
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # 这里的是否对起初的特征的进行下采样决定了是Bottleneck1还是Bottleneck2

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, # 不对我们最后一个BN层进行
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, # 不使用步幅来代替我们的扩张率
                 norm_layer=None):  # 设定参数，block是我们的Bottleneck模块，而layers是我们在 Bottleneck之前的两个模块
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # 设置BN层 

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]  # 这里的三个对应我们的扩张率都不会被去取代
        if len(replace_stride_with_dilation) != 3:  # 
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  #这个是根据通道来进行处理，当为一是所有输入均与所有输出进行卷积，当为2时，会并排设置两个卷积层来处理各自一半，然后再对结果融合、
        # 此外，等于channel时，就是对每一个通道进行处理，大小为输出通道除以输入通道
        self.base_width = width_per_group  #将基础宽度设置为64，图形设置成64的倍数，方便处理
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 输入通道为3 输出为64 卷积核大小为7 步幅为2，填充率为3，不加偏置
        self.bn1 = norm_layer(self.inplanes)  # 将输入通道传进去
        self.relu = nn.ReLU(inplace=True) # 和上面的一样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 进行最大池化，定义窗口大小，步幅为2，填充为一
        self.layer1 = self._make_layer(block, 64, layers[0])  # 使用50的话，layers的具体数值 [3, 4, 6, 3]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,  #使用101层的话，就是[3, 4, 23, 3]
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 将传入的特征图转变成1x1的大小
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 使用全连接层

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:  # 上面传进来的参数为不，就不会进入
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):  #构建layer
        norm_layer = self._norm_layer  # 设置BN层
        downsample = None  #不进行下采样
        previous_dilation = self.dilation # 扩张率
        if dilate:  #如果进行这一步，会将扩张率乘于步幅
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:  # 如果步幅不等于1或者设定输入维度不等于设定的扩张率4乘于传进来的通道数
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) # 构建下采样操作，也就是一个1x1卷积，和BN层

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))  #传入进去必要的形参
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))  # 这一步就是根据我们上面的layers的数目来构建Bottleneck的个数

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)  # 这是一个7x7的卷积，步幅为2，填充为3
        x = self.bn1(x)   # BN层
        x = self.relu(x)  # 使用RELU激活函数
        x = self.maxpool(x) #最大池化

        x = self.layer1(x)  # 构建layer1，传进去的通道数为64，使用了Bottleneck2，三个，不使用空洞卷积
        x = self.layer2(x)  # 在实际使用中，和layers1一样，只不过Bottleneck数目不同
        x = self.layer3(x)  # 在这一部分开始使用使用空洞卷积，根据步幅为2，默认的dilate为1以及第121代码可知，dilate为2，输入通道数为256，
        x = self.layer4(x)  # 和layer3一样，只不过数目不同，输入通道数为512

        x = self.avgpool(x)  # 进行自适应的平均池化操作
        x = torch.flatten(x, 1)  # 对特征图进行展平操作，从第一维开始
        x = self.fc(x)  #再进行一个全连接层的操作

        return x  # 构建出完整的架构

    def forward(self, x):
        return self._forward_impl(x)  # 前向传播的操作，将图片传进去，返回结果


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model  #构建模型，传进模块和layer以及一些参数


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)  # 通过这一步可以看出，我们的Bottleneck就是block
# 后面的对应的是我们layers的个数分布

def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
