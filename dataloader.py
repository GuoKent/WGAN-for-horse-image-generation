from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import warnings

warnings.filterwarnings("ignore")

# 数据预处理
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)


class my_dataset(Dataset):
    def __init__(self, store_path, split, data_transform=None):
        self.store_path = store_path
        self.split = split
        self.transforms = data_transform
        self.img_list = []  # 储存每张图片的路径
        self.label_list = []  # 储存每张图片的类别
        for file in glob.glob(self.store_path + '/' + split + '/*png'):
            cur_path = file.replace('\\', '/')  # 每张图片的路径,用/替代路径中的\\
            cur_label = cur_path.split('_')[-1].split('.png')[0]  # 获取每张图片的类别名
            self.img_list.append(cur_path)

    def __getitem__(self, item):
        img = Image.open(self.img_list[item]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)  # 转换
        return img

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':  # 测试用，无需运行
    store_path = './cifar'
    split = 'train'
    train_dataset = my_dataset(store_path, split, data_transform)
    dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=1)
    # print(type(train_dataset))

    ct = []
    # print(train_dataset.img_list)
    # print(train_dataset.label_list)
    for i, index in enumerate(train_dataset):  # 每次循环都会运行一次getitem函数，Dataset类规定好的
        data, label = index
        ct.append(label)
        print('data:', data)
        print('label:', label)
        break

    for batch_idx, (inputs, targets) in enumerate(dataset_loader):
        print(inputs)
        print(targets)
        # break
    '''for inputs, labels in dataset_loader:
        print(inputs, labels)'''
