from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImageDataset(Dataset):

    def __init__(self, classes:dict, img_path, transforms=None, num_class=100):
        self.classes = classes
        self.num_class = num_class
        self.img_path = img_path
        self.sizes = []
        for cls in sorted(classes.keys()):
            self.sizes.append(len(classes[cls]))
        self.length = sum(self.sizes)
        self.imgs = []
        for i, label in enumerate(sorted(self.classes.keys())):
            values = self.classes[label]
            for value in values:
                self.imgs.append({'label_i': i, 'img_path': value})
        self.transforms = transforms

    def read_img(self, path):
        img = plt.imread(self.img_path + path)
        img = Image.fromarray(img).convert('RGB')
        img = np.array(img).astype(np.uint8)
        return img

    def next(self, index):
        if index == self.__len__() - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(index+1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img1 = None
        label = None
        try:
            label = self.imgs[index]['label_i']
            img_path = self.imgs[index]['img_path']
            img1 = self.read_img(img_path)
            if self.transforms:
                img1 = self.transforms(img1)
        except Exception as e:
            print('Data corrupted. Index: ' + str(index))
            print(str(e))
            return self.next(index)
        return img1, label
