import torch
from fid_score.fid_score import FidScore

# 申明GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 计算FID分数
paths = ['./cifar10_horse/train', './cifar10_horse/WGAN_target32']
fid = FidScore(paths, device, batch_size=100)
score = fid.calculate_fid_score()
print('FIDScore:', score)
