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
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms  # 对传入的参数进行赋值

    def __getitem__(self, index):  #传入索引，返回图片和标签
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod  #静态方法，不需要实例化就可以调用
    def collate_fn(batch):  #对batch数据进行处理，将图片和标签分开
        images, targets = list(zip(*batch))
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
