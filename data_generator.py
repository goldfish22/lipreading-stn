import torch.utils.data as data
import glob,os
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as F
#b*c*t*h*w
class Grid_Datasetter_train(data.Dataset):
    def __init__(self,video_dir,time_step,max_seq_len,rotate=False):
        self.video_dir=video_dir
        self.videos=glob.glob(os.path.join(self.video_dir,'*/*'))
        self.video_num=len(self.videos)
        self.time_step=time_step
        self.train_transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
             ]
        )

        self.max_seq_len=max_seq_len
        self.rotate=rotate


    def __getitem__(self, item):
        path=self.videos[item]
        pics=[]
        angle=0
        if self.rotate==True:
            angle = random.uniform(-10, 10)
        for i in range(1,1+self.time_step):
            pic = os.path.join(path, '%d.jpeg' % i)
            pic = Image.open(pic).convert('RGB')#c*h*w
            if self.rotate==True:
                pic = F.rotate(pic, angle)
            pic = self.train_transform(pic)
            pics.append(pic)
        video = torch.stack(tuple(pics), dim=1)#cthw
        #video = torch.stack(tuple(pics), dim=1)#cthw
        # video = F1.interpolate(video, (32,64), mode='bilinear', align_corners=True)
        #
        # stn_img_feat, ctrl_points = self.stn(video)
        # video, _ = self.tps(video, ctrl_points)
        #video=torch.stack(tuple(pics),dim=1)#c*t*h*w
        #video=torch.transpose(video,0,1)
        inlabel, outlabel,label_len=self.convertToLabel(path.split('/')[-1])
        inlabel=torch.LongTensor(inlabel)
        outlabel = torch.LongTensor(outlabel)
        #label_len=torch.LongTensor((label_len,))
        return video,inlabel,outlabel,label_len

    def __len__(self):
        return self.video_num

    def convertToLabel(self, numbers):
        '''

        :param numbers: lgwl7a as str
        :return: lay green with ... as [12,1,25,27...]

        pad:0
        a-z:1-26
        space:27
        bos:28
        eos:29
        '''
        num2word = [{'l': 'lay', 's': 'set', 'b': 'bin', 'p': 'place'},
                    {'r': 'red', 'g': 'green', 'b': 'blue', 'w': 'white'},
                    {'w': 'with', 'i': 'in', 'a': 'at', 'b': 'by'},
                    {},
                    {'1': 'one',
                     '2': 'two',
                     '3': 'three',
                     '4': 'four',
                     '5': 'five',
                     '6': 'six',
                     '7': 'seven',
                     '8': 'eight',
                     '9': 'nine',
                     'z': 'zero'
                     },
                    {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}]
        str_num = []
        for idx, character in enumerate(numbers):
            if idx != 3:
                str_num.append(num2word[idx][character])
            else:
                str_num.append(character)
        str_num=' '.join(str_num)
        label = [ord(digit) - ord('a') + 1 if digit != ' ' else 27 for digit in str_num]
        inlabel = [28] + label + [29]
        outlabel = label + [29]
        inlabel_len = len(inlabel)
        outlabel_len = inlabel_len-1
        inlabel += [0] * (self.max_seq_len - inlabel_len)
        outlabel += [0] * (self.max_seq_len - outlabel_len)#modify

        return inlabel, outlabel, inlabel_len

class Grid_Datasetter_test(data.Dataset):
    def __init__(self,video_dir,time_step,max_seq_len):
        self.video_dir=video_dir
        self.videos=glob.glob(os.path.join(self.video_dir,'*/*'))
        self.video_num=len(self.videos)
        self.time_step=time_step

        self.train_transform=transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
             ]
        )
        self.max_seq_len=max_seq_len

    def __getitem__(self, item):
        path=self.videos[item]
        pics=[]
        #video=torch.Tensor(self.time_step,3,50,100)
        for i in range(1,1+self.time_step):
            pic=os.path.join(path,'%d.jpeg'%i)
            pic=Image.open(pic).convert('RGB')
            pic = self.train_transform(pic)

            #video[i-1]=pic
            pics.append(pic)
        #video=torch.stack(tuple(pics),dim=1)
        video=torch.stack(tuple(pics),dim=1)#ct*h*w

        label=self.convertToLabel(path.split('/')[-1])

        return video,label

    def __len__(self):
        return self.video_num

    def convertToLabel(self,numbers):
        '''

        :param numbers: lgwl7a as str
        :return: lay green with ... as [23,3,15,12...]

        pad:0
        a-z:1-26
        space:27
        bos:28
        eos:29
        '''
        num2word=[{'l':'lay','s':'set','b':'bin','p':'place'},
                    {'r':'red','g':'green','b':'blue','w':'white'},
                    {'w':'with','i':'in','a':'at','b':'by'},
                    {},
                    {'1':'one',
                    '2':'two',
                    '3':'three',
                    '4':'four',
                    '5':'five',
                    '6':'six',
                    '7':'seven',
                    '8':'eight',
                    '9':'nine',
                    'z':'zero'
                    },
                    {'a':'again','n':'now','p':'please','s':'soon'}]
        str_num=[]
        for idx,character in enumerate(numbers):
            if idx!=3:
                str_num.append(num2word[idx][character])
            else:
                str_num.append(character)
        return ' '.join(str_num)

class Vsa_Datasetter_train(data.Dataset):
    def __init__(self,video_dir,time_step,max_seq_len,rotate=False):
        self.video_dir=video_dir
        self.videos=glob.glob(os.path.join(self.video_dir,'*/*/*'))
        self.video_num=len(self.videos)
        self.time_step=time_step
        self.train_transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
             ]
        )
        self.max_seq_len=max_seq_len
        self.rotate = rotate

    def __getitem__(self, item):
        path=self.videos[item]
        pics=[]
        angle = 0
        if self.rotate == True:
            angle = random.uniform(-10, 10)
        for i in range(1,1+self.time_step):
            pic = os.path.join(path, '%03d.jpg' % i)
            pic = Image.open(pic).convert('RGB')#c*h*w
            if self.rotate == True:
                pic = F.rotate(pic, angle)
            pic = self.train_transform(pic)
            pics.append(pic)
        video=torch.stack(tuple(pics),dim=1)#c*t*h*w
        inlabel, outlabel,label_len=self.convertToLabel(path.split('/')[-1])
        inlabel=torch.LongTensor(inlabel)
        outlabel = torch.LongTensor(outlabel)
        #label_len=torch.LongTensor((label_len,))
        return video,inlabel,outlabel,label_len

    def __len__(self):
        return self.video_num

    def convertToLabel(self, numbers):
        '''

        :param numbers: lgwl7a as str
        :return: lay green with ... as [12,1,25,27...]

        pad:0
        a-z:1-26
        space:27
        bos:28
        eos:29
        '''
        num2word = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }
        str_num = ' '.join([num2word[digit] for digit in numbers])
        label = [ord(digit) - ord('a') + 1 if digit != ' ' else 27 for digit in str_num]
        inlabel = [28] + label + [29]
        outlabel = label + [29]
        inlabel_len = len(inlabel)
        outlabel_len = len(outlabel)
        inlabel += [0] * (self.max_seq_len - inlabel_len)
        outlabel += [0] * (self.max_seq_len - outlabel_len)
        return inlabel, outlabel, inlabel_len

class Vsa_Datasetter_test(data.Dataset):
    def __init__(self,video_dir,time_step):
        self.video_dir=video_dir
        self.videos=glob.glob(os.path.join(self.video_dir,'*/*/*'))
        self.video_num=len(self.videos)
        self.time_step=time_step
        self.train_transform=transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
             ]
        )

    def __getitem__(self, item):
        path=self.videos[item]
        pics=[]
        for i in range(1,1+self.time_step):
            pic=os.path.join(path,'%03d.jpg'%i)
            pic=Image.open(pic).convert('RGB')
            pic=self.train_transform(pic)
            pics.append(pic)
        video=torch.stack(tuple(pics),dim=1)
        label=self.convertToLabel(path.split('/')[-1])

        return video,label

    def __len__(self):
        return self.video_num

    def convertToLabel(self,numbers):
        '''

        :param numbers:6315
        :return: six three one five

        pad:0
        a-z:1-26
        space:27
        bos:28
        eos:29
        '''
        num2word = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }
        str_num = ' '.join([num2word[digit] for digit in numbers])
        return str_num