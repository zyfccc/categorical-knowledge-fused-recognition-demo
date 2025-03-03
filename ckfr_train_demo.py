import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
from libs.ImageDataset_train import ImageDataset
from libs.ImageDataset_val import ImageDataset as ImageDataset_test
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR


total_num_class = 100

TESTING = True
TRAINING = True
SAVING_MODEL = True


## A pretrained model file
PRETRAIN_FILENAME = 'model/pretrained.pth'
MODEL_FILENAME = 'model/ckfr.pth'
OPT_FILENAME = 'model/opt.pth'
LOG_FILENAME = 'model/log.txt'

FREEZE = False
SAVING_OPT = True

ALPHA = 20.0
L = 1.2

class ResNetWrapper(nn.Module):
    def __init__(self, resnet:torchvision.models.ResNet) -> None:
        super().__init__()
        self.resnet = resnet

    def forward(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        return self.classify(x)
    
    def encode(self, x: Tensor) -> Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def classify(self, x: Tensor) -> Tensor:
        x = self.resnet.fc(x)
        return x

def set_require_grad(model:nn.Module, state):
    for item in model.parameters():
        item.requires_grad = state

criterion = nn.CrossEntropyLoss()
def loss_function(pred, latents, latents2, latents3, label, dis2_label, dis3_label):
    label = label.long()
    one_hot = nn.functional.one_hot(label, num_classes=total_num_class)
    one_hot = one_hot.float()
    p_loss = criterion(pred, one_hot)

    latents_ = torch.flatten(latents, start_dim=1)
    latents2_ = torch.flatten(latents2, start_dim=1)
    latents3_ = torch.flatten(latents3, start_dim=1)
    latents_ = (latents_-torch.mean(latents_))/torch.std(latents_)
    latents2_ = (latents2_-torch.mean(latents2_))/torch.std(latents2_)
    latents3_ = (latents3_-torch.mean(latents3_))/torch.std(latents3_)

    lat_dis1 = torch.mean(torch.pow(torch.abs(latents2_-latents_), L), dim=1)
    lat_dis2 = torch.mean(torch.pow(torch.abs(latents3_-latents_), L), dim=1)
    dis_loss_ = torch.abs(lat_dis1*dis3_label-lat_dis2*dis2_label)
    dis_loss = torch.mean(dis_loss_) * ALPHA
    return p_loss, dis_loss


def train(train_loader, test_loader, epochs_num, lr, step_size):
    print('Hyperparameter alpha: ' + str(ALPHA))
    print('Hyperparameter ell: ' + str(L))

    resnet = torchvision.models.wide_resnet50_2()
    try:
        resnet.load_state_dict(torch.load(PRETRAIN_FILENAME))
        print('Pretrained model loaded.')
    except Exception as e:
        print(str(e))

    resnet.fc = nn.Linear(resnet.fc.in_features, total_num_class)
    model:ResNetWrapper = ResNetWrapper(resnet=resnet)
    
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME))
        print('Model loaded.')
    except Exception as e:
        print(str(e))

    if FREEZE:
        set_require_grad(model.resnet, False)
        set_require_grad(model.resnet.fc, True)
        print('Freeze mode.')
    else:
        set_require_grad(model.resnet, True)

    if torch.cuda.is_available():
        model.cuda()    

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

    if SAVING_MODEL:
        try:
            optimizer.load_state_dict(torch.load(OPT_FILENAME))
            print('Optimizer loaded.')
        except Exception as e:
            print(str(e))


    n = 0
    acc = 0
    for epoch in range(epochs_num):
        print('Epoch {}/{}'.format(epoch, epochs_num - 1))
        print('-'*10)

        model.train()

        if TRAINING:
            for i, data in enumerate(train_loader):
                torch.cuda.empty_cache()

                if data is None:
                    continue

                img1, img2, img3, label, dis2_label, dis3_label = data
                img1, img2, img3, label, dis2_label, dis3_label = img1.cuda(), img2.cuda(), img3.cuda(), label.cuda(), dis2_label.cuda(), dis3_label.cuda()

                dis2_label *= 0.1
                dis3_label *= 0.1

                optimizer.zero_grad()
                latents = model.encode(img1)
                latents2 = model.encode(img2)
                latents3 = model.encode(img3)
                pred_class = model.classify(latents)
                p_loss, dis_loss = loss_function(pred_class, latents, latents2, latents3, label, dis2_label, dis3_label)
                t_loss = p_loss + dis_loss
                t_loss.backward()
                optimizer.step()

                prediction = torch.max(pred_class, 1)[1]
                correct = torch.sum(prediction==label)
                acc += correct
                if (n + 1) % 20 == 0:
                    batch = pred_class.size(0)
                    
                    print('%d/%d p_loss: %.3f; accuracy: %.3f; dis_loss: %.3f;' %
                        (i+1, len(train_loader), p_loss/batch, acc/(batch*20), dis_loss/batch))
                    acc = 0

                if (n + 1) % 600 == 0:
                    print('saving model...')
                    torch.save(model.state_dict(), MODEL_FILENAME)
                    if SAVING_OPT:
                        torch.save(optimizer.state_dict(), OPT_FILENAME)

                n += 1
        
        scheduler.step()

        print('\n Testing ....')
        model.eval()  # For testing
        
        total_correct = 0
        total_count = 0

        test_loss_path = LOG_FILENAME
        for t_i, t_data in enumerate(test_loader):

            t_img, t_label = t_data
            t_inputs = Variable(t_img).cuda()
            t_label = Variable(t_label).cuda()
            with torch.no_grad():
                t_output = model(t_inputs)

            batch = t_output.size(0)
            prediction = torch.max(t_output, 1)[1]
            correct = torch.sum(prediction==t_label)
            total_count += batch
            total_correct += correct.item()

        with open(test_loss_path, 'a') as test_loss_f:
            test_loss_f.write('Total accuracy: ' + str(total_correct / total_count))
            test_loss_f.write('\n')
            test_loss_f.flush()

        torch.cuda.empty_cache()

    print('Finished Training and Testing')
    return model

def get_classes(csv_path):
    if isinstance(csv_path, list):
        records = None
        for item in csv_path:
            r = pd.read_csv(item)
            if records is None:
                records = r
            else:
                records = pd.concat([records, r], axis=0)
    else:
        records = pd.read_csv(csv_path)
    size = records.shape[0]
    dic = {}
    for i in range(size):
        key = records.iloc[i]['label']
        value = records.iloc[i]['filename']
        if key in list(dic.keys()):
            dic[key].append(value)
        else:
            dic[key] = [value]
    return dic


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default='../../dataset/mini-imagenet/images/', help='path to the image dataset')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1E-5,
                        help='learning rate')
    parser.add_argument('--step_size', type=int, default=30,
                        help='step size')
    
    parser.add_argument('--alpha', type=float, default=20.0,
                        help='Alpha hyperparameter of CKFR.')
    parser.add_argument('--ell', type=float, default=1.2,
                        help='L hyperparameter of CKFR.')

    args = parser.parse_args()
    img_path = args.img_path
    batch_size = args.batch_size
    epochs_num = args.epochs
    lr = args.lr
    step_size = args.step_size
    ALPHA = args.alpha
    L = args.ell

    train_paths = get_classes(['../../dataset/mini-imagenet/train.csv','../../dataset/mini-imagenet/test.csv','../../dataset/mini-imagenet/val.csv'])
    test_paths = train_paths.copy()
    
    indices = np.arange(len(list(train_paths.values())[0]))
    [indices_train, indices_valid] = train_test_split(indices, test_size=0.2, random_state=0, shuffle=True)
    
    for label in train_paths.keys():
        values = train_paths[label]
        train_paths[label] = list(np.array(values)[indices_train])
    for label in test_paths.keys():
        values = test_paths[label]
        test_paths[label] = list(np.array(values)[indices_valid])

    data_transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((256,256), scale=(0.9, 1.0), ratio=(0.75,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAutocontrast(),
        transforms.RandomRotation(10, interpolation=2),
        transforms.RandomApply([transforms.GaussianBlur((5,5), sigma=(0.6,1.5))], p=0.6),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
    ])
    
    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
    ])

    total_num_class = len(train_paths.values())

    train_dataset = ImageDataset(
        train_paths, img_path, data_transforms_train, total_num_class)
    test_dataset = ImageDataset_test(
        test_paths, img_path, data_transforms_test, total_num_class)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    trained_model = train(train_loader, test_loader, epochs_num, lr, step_size)
