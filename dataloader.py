import torch.utils.data as data
import os
import re
from scipy import misc
from PIL import Image
import numpy as np
# 이제 dataset
a = []
class dset(data.Dataset):
    """
    adversarial의 dVAE 를 막기 위한 dataset.
    image, fast gradient를 통해 공격 받은 image, target을 내보낸다.
    """
    def __init__(self,train_attack_folder_loc, transform = None, train = True):
        """
        args :
            train_attack_folder_loc(str) :  cifar10 50000줄의 dset에서 attack 받은 놈들이 들어있는 폴더 location
                 './attack/train/*' 형태로 들어와야 한다.
            transform (callable, optional): Optional transform to be applied
                on a sample. 본 transform은 attacked를 위한 transform이다.
                train(boolean) : true 면 train set, false 면 test set 이다.
        """
        # 원본 trainset 불러오기.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if train == True:
            self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        else:
            self.trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
        # attack data는 dictionary. index는 trainset[i]의 i 이고, value가 numpy이다.
        self.attack_data = {}
        # dictionary 채우기 시작train_attack_folder_loc
        # 일단 line 다 불러와from

        # path 수정
        if "*" in train_attack_folder_loc:
            passa = []
        else:
            train_attack_floder_loc = os.path.join(train_attack_floder_loc,'*')

        # list that has all the folders that has images right under
        train_attack_folder = glob.glob(train_attack_folder_loc)

        # file_lis if where all the image location is added
        file_lis = []
        for i in train_attack_folder:
            tmp = os.path.join(i,'*')
            file_lis = file_lis + glob.glob(tmp)

        if(len(file_lis) != len(self.trainset.train_labels)):
            print("your folder is somewhat wrong")
            return
        # now, load all the images and store it in attack_data dict
        com = re.compile(r'/([0-9]+).png')
        for file in file_lis:
            idx = int(com.search(file).group(1))
            data = pil_loader(file) # pil로 연다
            self.attack_data[idx] = data

        self.transform = transform

    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self,idx):
        """
        원본 img, attack img, target label을 내보냄.
        """
        global a
        img, target = self.trainset[idx]
        attacked = self.attack_data[idx]

        if self.transform:
            a = attacked
            attacked = self.transform(Image.fromarray(np.uint8(attacked)))
        return img, attacked, target
def _collate_fn(batch):
    """
    loader에서 output이 나올 함수.
    args: list. list of data, that is a tuple of (img,attacked,target)
    """
    minibatch_size = len(batch)
    img_out = batch[0][0].unsqueeze(0)
    attacked_out = batch[0][1].unsqueeze(0)
    target_out = [batch[0][2]]
    for i in range(1,minibatch_size):
        img_out = torch.cat((img_out,batch[i][0].unsqueeze(0)),0)
        attacked_out = torch.cat((attacked_out,batch[i][1].unsqueeze(0)),0)
        target_out.append(batch[i][2])
    return img_out, attacked_out, target_out

class loader(data.DataLoader):
    """
    loader class for adversarial defense.
    """
    def __init__(self,*args,**kwargs):
        super(loader,self).__init__(*args,**kwargs)
        self.collate_fn = _collate_fn
