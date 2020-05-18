import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

#CIFAR10をロードする関数
def load_CIFAR(resize, mean, std, batch_size):
    train_dataset = datasets.CIFAR10(
        "./data",
        train = True,
        #データオーギュメンテーション
        transform=transforms.Compose([
            transforms.RandomResizedCrop(resize, scale = (0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        download = True 
    )
    print(train_dataset)

    test_dataset = datasets.CIFAR10(
        "./data",
        train = False,
        transform=transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        download = True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle = True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle = False
    )

    return {'train': train_dataloader, 'test': test_dataloader}

size = 224
mean = (0.458, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

batch_size = 32

dataloaders_dict = load_CIFAR(resize = 224, mean = mean, std = std, batch_size = batch_size)

#ラベルとデータサイズ表示
'''
batch_iterator = iter(dataloaders_dict['train'])
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels)
'''

#画像表示
'''
for i in range(0, 5):
    img = inputs[i].numpy().transpose((1, 2, 0))
    print(img.shape)
    #img = np.squeeze(img)
    plt.imshow(img)
    plt.show()
'''
#学習済みのVGG-16モデルのロード
user_pretrained = True
net = models.vgg16(pretrained=user_pretrained)

#10クラス分類
net.classifier[6] = nn.Linear(in_features = 4096, out_features = 10)
print(net)

#訓練モードに設定
net.train()

#損失関数
criterion = nn.CrossEntropyLoss()

#転移学習のパラメータ設定
params_to_update = []

#学習させるパラメータ名
update_param_names = [
    "classifier.0.weight","classifier.0.bias",
    "classifier.3.weight","classifier.3.bias",
    "classifier.6.weight", "classifier.6.bias",
    ]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False
    
#print('---------')
#print(params_to_update)   

#最適化手法の設定
optimizer = optim.SGD(params = params_to_update, lr = 0.001, momentum = 0.9)

#モデルを学習させる関数
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    #損失値と認識率を保存するリスト
    history = {
        'train_loss': [],
        'train_acc' :[],
        'test_loss': [],
        'test_acc': []
    }

    #GPU初期設定

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス: ", device)

    #ネットワークをGPUへ
    net.to(device)
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print('Epoch {} / {}'.format(epoch + 1, num_epochs))
        print('----------')

        for phase in ['train', 'test']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            
            epoch_loss = 0.0
            epoch_corrects = 0
            
            #if (epoch == 0) and (phase == 'train'):
             #   continue
            

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                
                #GPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #勾配の初期化
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    plt.figure()
    plt.plot(range(1, num_epochs + 1, 1), history['train_loss'], label = 'train_loss')
    plt.plot(range(1, num_epochs + 1, 1), history['test_loss'], label = 'test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss_change_param.png')
    
    plt.figure()
    plt.plot(range(1, num_epochs + 1, 1), history['train_acc'], label = 'train_acc')
    plt.plot(range(1, num_epochs + 1, 1), history['test_acc'], label = 'test_acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('acc_change_param.png')

num_epochs = 20
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs = num_epochs)

                

