import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import my_dataset
import numpy as np
import torchvision.utils as utils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from WGAN import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter
from fid_score.fid_score import FidScore
from keras.preprocessing.image import array_to_img
from torchvision.utils import save_image


def initialize_weights(model):
    # 根据原论文,初始化模型参数
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # R,G,B每层的归一化用到的均值和方差
])

# 创建数据集
path = './cifar10_horse'
split = 'train'
train_dataset = my_dataset(path, split, transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)  # 每张图片32*32
# print(train_dataset[0])

# 申明GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

learning_rate = 0.00005
# image_size = 32
channels_image = 3
z_dim = 256
num_epoch = 200  # epoch过小训练不出来
feature_D = 64
feature_G = 64
weight_clip = 0.01

# 将输出还原32*32
resize = transforms.Resize(32)

# 定义并初始化模型参数
G = Generator(z_dim, channels_image, feature_G).to(device)
D = Discriminator(channels_image, feature_D).to(device)
initialize_weights(G)
initialize_weights(D)

# i定义优化器
opt_gen = optim.RMSprop(G.parameters(), lr=learning_rate)
opt_critic = optim.RMSprop(D.parameters(), lr=learning_rate)

# 开启训练模式
G.train()
D.train()

for epoch in range(num_epoch):
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        # 为了让noise和fake在for外面有效,需在这定义
        noise = 0
        fake = 0
        loss_critic = 0

        # 训练判别器
        for _ in range(5):
            noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
            fake = G(noise)
            critic_real = D(data).reshape(-1)
            critic_fake = D(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))  # 损失函数
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            # 将绝对值截断到(-0.01,0.01)
            for p in D.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # 训练生成器
        gen_fake = D(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        print(
            f"Epoch [{epoch + 1}/{num_epoch}] Batch {batch_idx + 1}/{len(train_loader)} "
            f"Loss D: {loss_critic:.4f}  Loss G: {loss_gen:.4f}")


# 反归一化函数
def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


with torch.no_grad():  # 取消梯度
    # 保存生成的1000张图片
    noise = torch.randn(1000, z_dim, 1, 1).to(device)  # 产生随机噪声
    fake_imgs = G(noise)  # 经过生成器生成图像
    # fake_imgs = G(noise).cpu().numpy()  # 生成1000张图像的集合(1000,3,32,32)
    # fake_imgs = D(fake_imgs)

    for img in range(fake_imgs.shape[0]):
        newImg = fake_imgs[img]  # (3,64,64)

        newImg = de_norm(newImg)

        # save_image(newImg, f'./cifar10_horse/target64/{img + 1}.png')  # 保存1000张图片,自带*255
        # newImg = array_to_img(newImg.transpose())  # transpose维度转化(3,32,32)->(32,32,3)
        # newImg = torch.from_numpy(newImg)
        newImg = resize(newImg)  # 还原回32*32
        save_image(newImg, f'./cifar10_horse/target/{img + 1}.png')  # 保存1000张图片,自带*255

# 保存模型
torch.save(G.state_dict(), 'Generator.pkl')

# 计算FID分数
paths = ['./cifar10_horse/train', './cifar10_horse/target']
fid = FidScore(paths, device, batch_size=100)
score = fid.calculate_fid_score()
print('FIDScore:', score)
