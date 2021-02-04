import os
import torch as t
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms as T
# list_path: ../data/lists/train.txt
# file_path: ../data/images/

normalize = T.Compose([T.Resize(256),
                       T.CenterCrop(256),
                       T.ToTensor(),
                      T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

def get_train_path(list_path, file_path):
    image = []
    label = []
    with open(list_path, "r") as lines:
        for line in lines:
            # line: 001.Black_footed_Albatross/Black_footed_Albatross_0004_2731401028.jpg\n
            line = line.rstrip('\n')
            img_pth = os.path.join(file_path, line)
            image.append(img_pth)
            label.append(int(line.split('.')[0])-1)
    return image, label

# 001.Black_footed_Albatross/Black_footed_Albatross_0004_2731401028.jpg
class cub_dataset(data.Dataset):
    # 自定义的参数
    def __init__(self, list_path, file_path,transforms=normalize,debug=False,test=False):
        images, labels = get_train_path(list_path, file_path)
        self.paths = images
        self.labels = labels
        self.transforms = transforms
        self.debug=debug
        self.test=test


    # 返回图片个数
    def __len__(self):
        return len(self.paths)


    # 获取每个图片
    def __getitem__(self, item):
        # path
        img_path =self.paths[item]
        # read image
        img = Image.open(img_path)
        # augmentation
        if self.transforms is not None:
            img = self.transforms(img)
        # read label
        label = self.labels[item]
        # return
        return img, int(label)

if __name__ == '__main__':
    # imgs, labels = get_train_path('../data/lists/train.txt', '../data/images/')
    dataset = cub_dataset('../data/lists/train.txt', '../data/images/')
    img, label = dataset.__getitem__(0)
    print(img, label)