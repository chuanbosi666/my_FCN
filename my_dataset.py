import os

import torch.utils.data as data
from PIL import Image

# 写入我们的数据集
class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')  # 图片的路径
        mask_dir = os.path.join(root, 'SegmentationClass')  # 掩码的路径

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)  # 图片索引的路径
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)  # 判断是否存在
        with open(os.path.join(txt_path), "r") as f:  # 打开文件
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]  #    读取文件中的内容，去掉空格

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]  # 将图片的路径和图片索引的文件名进行拼接
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]  # 将掩码的路径和图片索引的文件名进行拼接
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms  # 对传入的参数进行赋值

    def __getitem__(self, index):  #传入索引，返回图片和标签
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')  # 读取图片，转换为RGB格式
        target = Image.open(self.masks[index])  # 读取掩码

        if self.transforms is not None:
            img, target = self.transforms(img, target)  # 对图片和掩码进行变换

        return img, target

    def __len__(self):
        return len(self.images)  # 返回图片的长度

    @staticmethod  #静态方法，不需要实例化就可以调用
    def collate_fn(batch):  #对batch数据进行处理，将图片和标签分开
        images, targets = list(zip(*batch))  # 将batch数据分开，使用zip函数，zip函数是将可迭代对象打包成元组的列表
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值，这里的channel都为3
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images])) # 对图片的尺寸进行比较，取最大的尺寸，
    batch_shape = (len(images),) + max_size # 加上最大的尺寸，因为图片的尺寸不一样，所以要加上最大的尺寸
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value) # 生成一个新的tensor，大小为batch_shape，填充为0
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)  # 通过切片将图片的数据填充到batched_imgs中
    return batched_imgs  # 最后再返回


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
