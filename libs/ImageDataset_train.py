from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

CACHE_FILE = './pseudo_distances.npy'

class ImageDataset(Dataset):

    cache = {}
    try:
        cache = np.load(CACHE_FILE, allow_pickle=True).item()
    except Exception as e:
        print('cache file not exist.')
        print(str(e))

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
                self.imgs.append({'label_i': i, 'img_path': value, 'label': label})

        self.transforms = transforms
                
        
    def get_dis(self, label1, label2):
        if (label1+'-'+label2) not in ImageDataset.cache.keys():
            raise Exception(str((label1+'-'+label2)) + ' distance not found.')
        return ImageDataset.cache[(label1+'-'+label2)]

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
        label_i = None
        dis2 = float('nan')
        dis3 = float('nan')
        try:
            label_i = self.imgs[index]['label_i']
            label = self.imgs[index]['label']
            img_path = self.imgs[index]['img_path']
            
            randi = np.random.randint(0, self.length)
            img_path2 = self.imgs[randi]['img_path']
            label2 = self.imgs[randi]['label']

            randc = np.random.randint(0, self.length)
            img_path3 = self.imgs[randc]['img_path']
            label3 = self.imgs[randi]['label']
            
            dis2 = self.get_dis(label, label2)
            dis3 = self.get_dis(label, label3)

            img1 = self.read_img(img_path)
            img2 = self.read_img(img_path2)
            img3 = self.read_img(img_path3)

            if self.transforms:
                img1 = self.transforms(img1)
                img2 = self.transforms(img2)
                img3 = self.transforms(img3)

        except Exception as e:
            print('Data corrupted. Index: ' + str(index))
            print(str(e))
            return self.next(index)

        return img1, img2, img3, label_i, dis2, dis3
