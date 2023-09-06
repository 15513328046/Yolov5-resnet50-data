import math
import os.path
import torch
from timm.loss import LabelSmoothingCrossEntropy
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets, models
import time
import numpy as np
import random
from PIL import Image, ImageEnhance
import timm

from utils import Resize_with_Ratio, MyImageFolder, Cutout

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean, std = (0.42760366, 0.4122089, 0.38598374), (0.17665266, 0.17428997, 0.16621284)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(net, train_loader, test_loader, criterion, optimizer, num_epochs, writer=None, save_dir=''):
    net = net.to(device)
    print("training on", device)
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # 训练模式
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # 梯度清零
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        save_dir_2 = './output/{}'.format(save_dir)
        if not os.path.exists(save_dir_2):
            os.mkdir(save_dir_2)
        torch.save(net.state_dict(), '{save_dir}/snapshot_{epoch}.pth'.format(
            save_dir=save_dir_2,
            epoch=epoch + 1
        ))

        err_imgs = []
        with torch.no_grad():
            net.eval()  # 评估模式
            test_acc_sum, n2 = 0.0, 0
            for X, y, img_paths in test_loader:
                out = net(X.to(device))
                p_y = out.argmax(dim=1)
                err_ind = np.where((p_y != y.to(device)).cpu())
                if len(err_ind[0]):
                    err_img_paths = (np.array(img_paths)[err_ind[0]]).tolist()
                    err_cls = map(lambda x: train_loader.dataset.classes[x], np.array(p_y.cpu())[err_ind[0]].tolist())

                    err_imgs.extend(list(zip(err_img_paths, err_cls)))

                test_acc_sum += (p_y == y.to(device)).float().sum().cpu().item()
                n2 += y.shape[0]

        test_acc_avg = test_acc_sum / n2
        if test_acc_avg > 0.92:
            print('infer wrong imgs: {}'.format(err_imgs))
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec\r\n'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_acc_avg, time.time() - start))
        writer.add_scalar(f'test_acc', test_acc_avg, epoch)
        writer.add_scalar(f'train_acc', train_acc_sum / n, epoch)
        writer.add_scalar(f'train_loss', train_loss_sum / batch_count, epoch)


def main(train_dir, test_dir, batch_size, lr, num_epochs, model_name):
    train_augs = transforms.Compose([
        Resize_with_Ratio((300, 300)),
        transforms.RandomResizedCrop(size=224),
        # transforms.Resize(size=224),
        # transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(180),  # 旋转 -15~15
        # degrees 旋转角度   translate 水平和垂直平移的最大绝对偏移量 scale: 缩放比例  shear:放射变换角度
        # transforms.RandomAffine(degrees=0, translate=(0, 0), scale=(0.5, 0.9), shear=50),
        # transforms.RandomAffine(degrees=0, translate=(0, 0), shear=50),
        # transforms.ColorJitter(brightness=0.6, contrast=0.7, saturation=0.5, hue=0.1),
        transforms.ColorJitter(hue=[-0.1, 0.2]),
        # RandomContrast(2.8, 2.9),
        # RandomBrightness(1.1, 1.2),
        # RandomColor(2.3, 2.4),

        # rand_augment_transform(auto_augment_config, aa_params),
        transforms.ToTensor(),

        transforms.Normalize(mean, std),
        Cutout(2, 24),
        # RandomErasing(mean=mean),
    ])

    test_augs = transforms.Compose([
        Resize_with_Ratio((300, 300)),
        # transforms.Resize(size=300),  # 256
        transforms.CenterCrop(size=224),
        # RandomContrast(2.8, 2.9),
        # RandomBrightness(1.1, 1.2),
        # RandomColor(2.3, 2.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = datasets.ImageFolder(train_dir, transform=train_augs)
    test_set = MyImageFolder(test_dir, transform=test_augs)
    print(train_set.classes)  #

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    pretrained_net = timm.create_model(model_name, pretrained=True, num_classes=len(train_set.classes))

    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    optimizer = optim.SGD([
        {'params': feature_params},
        {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
        lr=lr,
        weight_decay=0.001,
    )

    loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    save_dir = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '_' + f'{model_name}'
    writer = SummaryWriter('tensorboard/{}'.format(save_dir))
    train(pretrained_net, train_loader, test_loader, loss, optimizer, num_epochs, writer=writer,
          save_dir=save_dir)


if __name__ == '__main__':
    train_dir = r'/home/xlz/Desktop/project/地面装备/data/cls_data/train_test/train'
    test_dir = r'/home/xlz/Desktop/project/地面装备/data/cls_data/train_test/test'
    batch_size = 32
    lr = 0.01
    num_epochs = 200
    setup_seed(0)
    model_name = 'resnet50'
    main(train_dir, test_dir, batch_size, lr, num_epochs, model_name)
