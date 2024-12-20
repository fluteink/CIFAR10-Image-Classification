import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from vit_pytorch import ViT, SimpleViT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # 加载和预处理数据集
    trans_train = transforms.Compose(
        [transforms.RandomResizedCrop(224),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
         # （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.08, 1.0)
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    trans_valid = transforms.Compose(
        [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
         transforms.CenterCrop(224),  # 依据给定的size从中心裁剪
         transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc

    trainset = torchvision.datasets.CIFAR10(root="./cifar10", train=True, download=True, transform=trans_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=False, transform=trans_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 随机获取部分训练数据
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # 显示图像
    # imshow(torchvision.utils.make_grid(images[:4]))
    # 打印标签
    print(''.join('%5s ' % classes[labels[j]] for j in range(4)))

    # 使用预训练模型
    model = get_vit_model()

    # 使用 DataParallel 包装模型以支持多 GPU
    #model = nn.DataParallel(model)  # 自动使用所有可用的 GPU
    #model = model.to(device)

    # 查看总参数及训练参数
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数:{}'.format(total_params))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # 只需要优化最后一层参数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3, momentum=0.9)  # 优化器

    # train
    train(model, trainloader, testloader, 30, optimizer, criterion)


# 计算准确率
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


# 显示图片
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 定义训练函数
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            label = label.to(device)  # (bs, h, w)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)  # (bs, 3, h, w)
                label = label.to(device)  # (bs, h, w)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


def get_vit_model():
    v = SimpleViT(
        image_size=224,
        patch_size=32,
        num_classes=10,
        dim=256,
        depth=2,
        heads=4,
        mlp_dim=128
    )
    return v


if __name__ == '__main__':
    main()

