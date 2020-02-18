import torch.utils.data as data
import glob,os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.nn import functional as F1
from torch.nn import Transformer,BatchNorm3d,AdaptiveMaxPool2d,AdaptiveAvgPool2d,BatchNorm1d,LayerNorm,Embedding,Parameter,ModuleList,Softmax,Sequential,Linear,Dropout3d,Module,Conv3d,Conv2d,Conv1d,MaxPool3d,BatchNorm2d,ReLU,MaxPool2d
import torch.nn as nn
import torch.optim as optim
from utils import calculateWer,AverageMeter,get_time,Spell
from tqdm import tqdm

from data_generator import Grid_Datasetter_train,Grid_Datasetter_test
from transformer_model import Transformer_model
# class Grid_Datasetter_train(data.Dataset):
#     def __init__(self,video_dir,time_step,max_seq_len,rotate=False):
#         self.video_dir=video_dir
#         self.videos=glob.glob(os.path.join(self.video_dir,'*/*'))
#         self.video_num=len(self.videos)
#         self.time_step=time_step
#         self.train_transform=transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#              ]
#         )
#
#         self.max_seq_len=max_seq_len
#         self.rotate=rotate
#
#
#     def __getitem__(self, item):
#         path=self.videos[item]
#         pics=[]
#         angle=0
#         if self.rotate==True:
#             angle = random.uniform(-10, 10)
#         for i in range(1,1+self.time_step):
#             pic = os.path.join(path, '%d.jpeg' % i)
#             pic = Image.open(pic).convert('RGB')#c*h*w
#             if self.rotate==True:
#                 pic = F.rotate(pic, angle)
#             pic = self.train_transform(pic)
#             pics.append(pic)
#         video = torch.stack(tuple(pics), dim=1)#cthw
#         #video = torch.stack(tuple(pics), dim=1)#cthw
#         # video = F1.interpolate(video, (32,64), mode='bilinear', align_corners=True)
#         #
#         # stn_img_feat, ctrl_points = self.stn(video)
#         # video, _ = self.tps(video, ctrl_points)
#         #video=torch.stack(tuple(pics),dim=1)#c*t*h*w
#         #video=torch.transpose(video,0,1)
#         inlabel, outlabel,label_len=self.convertToLabel(path.split('/')[-1])
#         inlabel=torch.LongTensor(inlabel)
#         outlabel = torch.LongTensor(outlabel)
#         #label_len=torch.LongTensor((label_len,))
#         return video,inlabel,outlabel,label_len
#
#     def __len__(self):
#         return self.video_num
#
#     def convertToLabel(self, numbers):
#         '''
#
#         :param numbers: lgwl7a as str
#         :return: lay green with ... as [12,1,25,27...]
#
#         pad:0
#         a-z:1-26
#         space:27
#         bos:28
#         eos:29
#         '''
#         num2word = [{'l': 'lay', 's': 'set', 'b': 'bin', 'p': 'place'},
#                     {'r': 'red', 'g': 'green', 'b': 'blue', 'w': 'white'},
#                     {'w': 'with', 'i': 'in', 'a': 'at', 'b': 'by'},
#                     {},
#                     {'1': 'one',
#                      '2': 'two',
#                      '3': 'three',
#                      '4': 'four',
#                      '5': 'five',
#                      '6': 'six',
#                      '7': 'seven',
#                      '8': 'eight',
#                      '9': 'nine',
#                      'z': 'zero'
#                      },
#                     {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}]
#         str_num = []
#         for idx, character in enumerate(numbers):
#             if idx != 3:
#                 str_num.append(num2word[idx][character])
#             else:
#                 str_num.append(character)
#         str_num=' '.join(str_num)
#         label = [ord(digit) - ord('a') + 1 if digit != ' ' else 27 for digit in str_num]
#         inlabel = [28] + label + [29]
#         outlabel = label + [29]
#         inlabel_len = len(inlabel)
#         outlabel_len = inlabel_len-1
#         inlabel += [0] * (self.max_seq_len - inlabel_len)
#         outlabel += [0] * (self.max_seq_len - outlabel_len)#modify
#
#         return inlabel, outlabel, inlabel_len
#
# class Grid_Datasetter_test(data.Dataset):
#     def __init__(self,video_dir,time_step,max_seq_len):
#         self.video_dir=video_dir
#         self.videos=glob.glob(os.path.join(self.video_dir,'*/*'))
#         self.video_num=len(self.videos)
#         self.time_step=time_step
#
#         self.train_transform=transforms.Compose(
#             [
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#              ]
#         )
#         self.max_seq_len=max_seq_len
#
#     def __getitem__(self, item):
#         path=self.videos[item]
#         pics=[]
#         #video=torch.Tensor(self.time_step,3,50,100)
#         for i in range(1,1+self.time_step):
#             pic=os.path.join(path,'%d.jpeg'%i)
#             pic=Image.open(pic).convert('RGB')
#             pic = self.train_transform(pic)
#
#             #video[i-1]=pic
#             pics.append(pic)
#         #video=torch.stack(tuple(pics),dim=1)
#         video=torch.stack(tuple(pics),dim=1)#ct*h*w
#
#         label=self.convertToLabel(path.split('/')[-1])
#
#         return video,label
#
#     def __len__(self):
#         return self.video_num
#
#     def convertToLabel(self,numbers):
#         '''
#
#         :param numbers: lgwl7a as str
#         :return: lay green with ... as [23,3,15,12...]
#
#         pad:0
#         a-z:1-26
#         space:27
#         bos:28
#         eos:29
#         '''
#         num2word=[{'l':'lay','s':'set','b':'bin','p':'place'},
#                     {'r':'red','g':'green','b':'blue','w':'white'},
#                     {'w':'with','i':'in','a':'at','b':'by'},
#                     {},
#                     {'1':'one',
#                     '2':'two',
#                     '3':'three',
#                     '4':'four',
#                     '5':'five',
#                     '6':'six',
#                     '7':'seven',
#                     '8':'eight',
#                     '9':'nine',
#                     'z':'zero'
#                     },
#                     {'a':'again','n':'now','p':'please','s':'soon'}]
#         str_num=[]
#         for idx,character in enumerate(numbers):
#             if idx!=3:
#                 str_num.append(num2word[idx][character])
#             else:
#                 str_num.append(character)
#         return ' '.join(str_num)

# class BasicBlock3D(Module):
#     '''
#     A BasicBlock for Resnet which passes through 2 conv3d-bn-relu modules and
#     adds residual unit on itself.
#     channel:inplanes->outplanes.
#     '''
#     def __init__(self,inplanes,outplanes,downsample=None,stride=1):
#         super(BasicBlock3D,self).__init__()
#         self.conv1=Conv3d(inplanes,outplanes,kernel_size=(1,3,3),stride=stride,padding=(0,1,1),bias=True)
#         self.bn1=BatchNorm3d(outplanes)
#         self.relu=ReLU(inplace=True)
#         self.conv2=Conv3d(outplanes,outplanes,kernel_size=(1,3,3),stride=stride,padding=(0,1,1),bias=True)
#         self.bn2=BatchNorm3d(outplanes)
#         self.downsample=downsample
#
#     def forward(self, x):
#         identity=x
#         x=self.conv1(x)
#         x=self.bn1(x)
#         x=self.relu(x)
#         x=self.conv2(x)
#         x=self.bn2(x)
#         if self.downsample is not None:#identity!identity!
#             identity=self.downsample(identity)
#         x+=identity
#         x=self.relu(x)
#
#         return x

# class Convolutional_Feature_Extractor(Module):
#     '''
#     channel:3->512
#     '''
#     def __init__(self,d_model):
#         super(Convolutional_Feature_Extractor,self).__init__()
#         #(N,Cin,D,H,W)
#         self.conv3d1=Conv3d(in_channels=3,out_channels=32,kernel_size=(5,5,5),padding=(2,2,2)
#                            # ,stride=(1,2,2)
#         )
#         self.mp3d1 = MaxPool3d(kernel_size=(1, 2, 2))
#         self.bn1=BatchNorm3d(32)
#         self.basicblock1 = BasicBlock3D(inplanes=32,outplanes=32)
#
#         self.conv3d2=Conv3d(in_channels=32,out_channels=64,kernel_size=(5,5,5),padding=(2,2,2)
#                            # ,stride=(1,2,2)
#                             )
#         self.mp3d2 = MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))#(N,Cin,D,H',W')
#         self.bn2 = BatchNorm3d(64)
#         self.basicblock2 = BasicBlock3D(inplanes=64, outplanes=64)
#
#         self.conv3d3 = Conv3d(in_channels=64, out_channels=96, kernel_size=(1, 5, 5), padding=(0, 2, 2)
#                              # ,stride=(1,2,2)
#                               )
#         self.mp3d3 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (N,Cin,D,H',W')
#         self.bn3 = BatchNorm3d(96)
#         self.basicblock3 = BasicBlock3D(inplanes=96, outplanes=96)
#
#         self.gap=AdaptiveAvgPool2d((1,1))
#         #self.linear=Linear(in_features=96*72,out_features=d_model)
#         self.linear = Linear(in_features=96, out_features=d_model)
#         self.bn=BatchNorm1d(d_model)
#
#     def forward(self, x):
#         #print (x)
#         x=self.conv3d1(x)
#         x=self.mp3d1(x)
#
#         #x=self.bn1(x)
#         x=self.basicblock1(x)
#
#         x=self.conv3d2(x)
#         x=self.mp3d2(x)
#         #x = self.bn2(x)
#         x = self.basicblock2(x)
#
#         x = self.conv3d3(x)
#         x = self.mp3d3(x)
#         #x = self.bn3(x)
#         x = self.basicblock3(x) #(N,Cin,D,6,12)
#         x=torch.transpose(x,1,2) #(N,D,C,H',W')
#         batch=x.size(0)
#         time_step=x.size(1)
#         x=x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)) #(N*T,C,H,W)          #(N,75,96*72)
#         x=self.gap(x)#(N*T,C,1,1)
#         x=x.squeeze(-1).squeeze(-1)#(N*T,96)
#         x = self.linear(x)     #(N*T,512)                                                 # (N,75,512)
#         x=x.view(batch,time_step,-1)#(N,T,512)
#         x=torch.transpose(x,1,2)#(N,512,75)
#
#         x=self.bn(x)
#         x = torch.transpose(x, 1, 2)#(N,75,512)
#         #assert x.size()==(BATCH_SIZE,TIME_STEP,d_model)
#         return x

# class Convolutional_Feature_Extractor_lipnet(Module):
#     '''
#     channel:3->512
#     '''
#     def __init__(self,d_model):
#         super(Convolutional_Feature_Extractor_lipnet,self).__init__()
#         #(N,Cin,D,H,W)
#         self.conv3d1=Conv3d(in_channels=3,out_channels=32,kernel_size=(3,5,5),stride=(1,2,2),padding=(1,2,2)
#                             )
#         self.bn1 = BatchNorm3d(32)
#         self.relu1=ReLU(inplace=True)
#         self.drop1=Dropout3d(0.5)
#         self.mp1=MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2))
#         #add
#         #self.basicblock1 = BasicBlock3D(inplanes=32, outplanes=32)
#
#         self.conv3d2=Conv3d(in_channels=32,out_channels=64,kernel_size=(3,5,5),stride=(1,1,1),padding=(1,2,2))
#         self.bn2 = BatchNorm3d(64)
#         self.relu2 = ReLU(inplace=True)
#         self.drop2=Dropout3d(0.5)
#         self.mp3d2 = MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))#(N,Cin,D,H',W')
#         #add
#         #self.basicblock2 = BasicBlock3D(inplanes=64, outplanes=64)
#
#         self.conv3d3 = Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), padding=(1, 1, 1))
#         self.bn3 = BatchNorm3d(96)
#         self.relu3 = ReLU(inplace=True)
#         self.drop3 = Dropout3d(0.5)
#         self.mp3d3 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (N,Cin,D,H',W')
#         #add
#         #self.basicblock3 = BasicBlock3D(inplanes=96, outplanes=96)
#
#         self.conv3d4=Conv3d(in_channels=96, out_channels=d_model,kernel_size=(1,1,1),stride=(1,1,1))
#         self.bn4 = BatchNorm3d(d_model)
#         #self.gmp=AdaptiveMaxPool2d((1,1))
#         self.gap=AdaptiveAvgPool2d((1,1))
#         self.linear=Linear(in_features=96,out_features=d_model)
#         self.bn = BatchNorm1d(d_model)
#
#     def forward(self, x):
#         x=self.conv3d1(x)
#         x=self.bn1(x)
#         x=self.relu1(x)
#         x = self.drop1(x)
#         x=self.mp1(x)
#         #add
#         #x = self.basicblock1(x)
#
#         x=self.conv3d2(x)
#         x=self.bn2(x)
#         x=self.relu2(x)
#         x=self.drop2(x)
#         x=self.mp3d2(x)
#         #add
#         #x = self.basicblock2(x)
#
#         x = self.conv3d3(x)
#         x=self.bn3(x)
#         x=self.relu3(x)
#         x = self.drop3(x)
#         x = self.mp3d3(x)
#         #add
#         #x = self.basicblock3(x)
#
#         #x=self.conv3d4(x)
#         #bn_add
#         #x=self.bn4(x)
#
#         x = torch.transpose(x, 1, 2)  # (N,D,C,H',W')
#         batch = x.size(0)
#         time_step = x.size(1)
#         x = x.contiguous().view(batch * time_step, x.size(2), x.size(3), x.size(4))#(ND,C,H,W)
#
#         #x=self.gmp(x)#(ND,C,1,1)
#         x=self.gap(x)#(ND,C,1,1)
#         x=x.squeeze(-1).squeeze(-1)#(ND,C)
#         x=self.linear(x)#(ND,D_MODEL)
#         x=x.view(batch,time_step,-1)#(N,T,C)
#
#         x=torch.transpose(x,1,2)#N,C,T
#         x=self.bn(x)
#         x=torch.transpose(x,1,2)#N,T,C
#         assert x.size()==(BATCH_SIZE,TIME_STEP,d_model)
#         return x
#
# class PositionalEncoding(Module):
#     '''
#         positional encoding for the series.
#         input:b*max_seq_len,each value ranges from [0,dictionary_size].
#         e.g.[1,2,3,4,5,0,0,0,0,0]
#         output:b*max_seq_len*d_mode;
#     '''
#     def __init__(self,max_seq_len,d_model):
#         super(PositionalEncoding,self).__init__()#dic_size=max_seq_len+1
#         self.posembedding=Embedding(max_seq_len+1,d_model)    #dictionary_size*d_model
#         posencoding=np.array(
#             [[t/np.power(10000,2.0*(d//2)/d_model) for d in range(d_model)]for t in range(1,1+max_seq_len)]
#         )#seq_len*d_model
#         posencoding[:,0::2]=np.sin(posencoding[:,0::2])
#         posencoding[:,1::2]=np.cos(posencoding[:,1::2])
#         posencoding=torch.Tensor(posencoding)
#         posencoding=torch.cat((torch.zeros([1,d_model]),posencoding))
#         self.posembedding.weight=Parameter(posencoding,requires_grad=False)
#
#     def forward(self,x):#x:batch*seq_len
#         output_pos=self.posembedding(x)
#         return output_pos#b*t(l)*d
#
# class I_hope_it_work(Module):
#     def __init__(self,d_model=512,vocabulary_size=30,max_seq_len=75,decoder_max_seq_len=35,
#                  nhead=8, num_encoder_layers=6,
#                  num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
#                  activation="relu",cf=False,stn_on=True,bs=16,ts=75):
#         super(I_hope_it_work,self).__init__()
#         self.cf=cf
#         self.stn_on=stn_on
#         if cf==True:
#             self.cfe=Convolutional_Feature_Extractor_lipnet(d_model)#(n,t,e)
#         else:
#             self.cfe=Convolutional_Feature_Extractor(d_model)#(n,t,e)
#         self.positionEncoding=PositionalEncoding(max_seq_len,d_model)
#         self.decodeEmbedding=Embedding(vocabulary_size,d_model)
#         self.decoder_max_seq_len=decoder_max_seq_len
#         self.transformer=Transformer(d_model=d_model, nhead=nhead,
#             num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,
#                 dim_feedforward=dim_feedforward, dropout=dropout,activation=activation)
#         self.linear=Linear(d_model,vocabulary_size)
#         self.ln=LayerNorm(d_model)
#         self.stn = STNHead(in_planes=3, num_ctrlpoints=20,
#                            activation=None)
#         self.tps = TPSSpatialTransformer(output_image_size=(50, 100),
#                                          num_control_points=20,
#                                          margins=tuple([0.05, 0.05]))
#         self.bs=bs
#         self.ts=ts
#         #self.softmax=Softmax(dim=-1)!!!!!no!
#
#     def forward(self,videos,labels,labels_len,tgt_mask,tgt_key_padding_mask=None):#videos:(N,Cin,D,H,W) label:(N,SEQ_LEN)
#         #videos:bcthw
#
#         if self.stn_on:
#             print ('stn')
#             videos=torch.transpose(videos,1,2)
#             #print (videos.size())
#             videos=videos.reshape(self.bs*self.ts,3,50,100)
#             stn_input = F1.interpolate(videos, (32, 64), mode='bilinear', align_corners=True)#(bt)c3264
#             stn_img_feat, ctrl_points = self.stn(stn_input)#(bt)n 2
#             videos, _ = self.tps(videos, ctrl_points)
#             videos=videos.view(self.bs,self.ts,3,50,100)
#             videos=torch.transpose(videos,1,2)#bcthw
#         src=self.cfe(videos)#(N,T,E)
#         encoder_pos = torch.LongTensor([[t for t in range(1, 1 + src.size(1))] for b in range(src.size(0))]).cuda()#(N,T)
#         encoding_pos = self.positionEncoding(encoder_pos)#(N,T,E)
#         src+=encoding_pos#(N,T,E)
#         src=torch.transpose(src,0,1)#(T,N,E)
#         tgt=self.decodeEmbedding(labels)#(N,SEQ_LEN,E)
#         decoder_pos = torch.LongTensor([[i for i in range(1, l.item() + 1)] + [0] * (self.decoder_max_seq_len - l.item()) for l in labels_len]).cuda()#(B,SEQ_LEN)
#         decoding_pos=self.positionEncoding(decoder_pos)
#         tgt+=decoding_pos#(N,SEQ_LEN,E)
#         tgt=torch.transpose(tgt,0,1)#(SEQ_LEN,N,E)
#         tranoutput=self.transformer(src,tgt,tgt_mask=tgt_mask
#                                     ,tgt_key_padding_mask=tgt_key_padding_mask)#(seq_len,N,E)
#         tranoutput=torch.transpose(tranoutput,0,1)#(N,seq_len,E)
#         tranoutput=self.ln(tranoutput)
#         tranoutput=self.linear(tranoutput)#(N,seq_len,voc_size)
#         #output=self.softmax(tranoutput)#(N,seq_len,voc_size)
#         output=tranoutput
#         return output

# def schedule_lr(epoch,optimizer):
#     if epoch==0:
#         for params in optimizer.param_groups:
#             params['lr'] =1e-4
#     if epoch==15:
#         for params in optimizer.param_groups:
#             params['lr'] =5e-5
#     if epoch==25:
#         for params in optimizer.param_groups:
#             params['lr'] =1e-5

if __name__=='__main__':

    pid = os.getpid()
    # f = open('pid_grid.log', 'w')
    # f.write(str(pid))
    # f.close()
    spell = Spell('grid.txt')
    BATCH_SIZE = 16#woc....increase it will cause CUDNN_STATUS_MAPPING_ERROR...WOCAO
    TIME_STEP = 75
    DECODER_MAX_SEQ_LEN = 35
    d_model=512
    VOCABULARY_SIZE=30
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    MAX_SEQ_LEN=max(TIME_STEP,DECODER_MAX_SEQ_LEN)
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    #remark="unseenSTN"
    remark=""
    seen = 'unseen'
    stn_on=False
    #stn_on=True
    #rotate = True
    rotate = False
    init_epoch=0
    ckpt = None
    #ckpt='trancheckpoint/kernel551-unseenSTN-866adam_Time2019-12-31-14-31_epoch26_loss0.019'
    #ckpt='trancheckpoint/kernel551-seenR-866adam_Time2019-12-16-13-38_epoch82_loss0.012_val0.034'

    train_mode = True
    val_mode = False
    val_once = False
    test_mode= False
    test_once=False
    save_ckpt=True

    if seen=='seen':
        train_ds = Grid_Datasetter_train('/ssddata/jinyu/lips_grid_seen/train',
                                         time_step=TIME_STEP, max_seq_len=DECODER_MAX_SEQ_LEN
                                         ,rotate=rotate
                                         )
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, pin_memory=True,num_workers=3)

        val_ds = Grid_Datasetter_test('/ssddata/jinyu/lips_grid_seen/val', time_step=TIME_STEP, max_seq_len=MAX_SEQ_LEN)
        val_loader = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     drop_last=True, pin_memory=True,num_workers=3)
    elif seen=='unseen':
        train_ds = Grid_Datasetter_train('/ssddata/jinyu/lips_grid_unseen/train',
                                         time_step=TIME_STEP,
                                         max_seq_len=DECODER_MAX_SEQ_LEN
                                         ,rotate=rotate
                                        )
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, pin_memory=True
                                       ,num_workers=3
                                       )
        test_ds = Grid_Datasetter_test('/ssddata/jinyu/lips_grid_unseen/test', time_step=TIME_STEP, max_seq_len=MAX_SEQ_LEN)
        test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      drop_last=True, pin_memory=True,num_workers=3)
    elif seen=='all':
        train_ds = Grid_Datasetter_train('/ssddata/jinyu/lips_grid_all/train', time_step=TIME_STEP,
                                         max_seq_len=DECODER_MAX_SEQ_LEN)
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, pin_memory=True)
    #val_ds = Grid_Datasetter_test('/ssddata/jinyu/lips_grid/train', time_step=TIME_STEP, max_seq_len=MAX_SEQ_LEN)
    #val_loader = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
     #                            drop_last=True, pin_memory=True)
    model=Transformer_model(d_model,vocabulary_size=VOCABULARY_SIZE,max_seq_len=MAX_SEQ_LEN,
                         decoder_max_seq_len=DECODER_MAX_SEQ_LEN, nhead = nhead,
                         num_encoder_layers = num_encoder_layers,
                         num_decoder_layers = num_decoder_layers,dim_feedforward = dim_feedforward,
                         dropout = dropout,activation = activation
                         ,stn_on=stn_on,bs=BATCH_SIZE,ts=TIME_STEP
                         )
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    #model = nn.DataParallel(model, device_ids=[0,2])
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(DEVICE)
    #LOSS=nn.CrossEntropyLoss(ignore_index=0)
    LOSS=nn.CrossEntropyLoss()
    OPTIMIZER=optim.Adam(model.parameters(),lr=1e-4)
    #OPTIMIZER = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    #OPTIMIZER = optim.Adadelta(model.parameters(),lr=1e-4)

    if not os.path.isdir('trancheckpoint'):
        os.mkdir('trancheckpoint')

    for epoch in range(200):
        """
        tgt_mask:like this below,will be added to attention after Q(dotproduct)KT when decoding
        [[0., -inf, -inf, -inf, -inf],
        [0.,   0.,  -inf, -inf, -inf],
        [0.,   0.,   0.,  -inf, -inf],
        [0.,   0.,   0.,   0.,  -inf],
        [0.,   0.,   0.,   0.,   0.]]
        """
        tgt_mask = torch.triu(torch.ones((DECODER_MAX_SEQ_LEN, DECODER_MAX_SEQ_LEN), dtype=torch.float) * float('-inf'),
                              diagonal=1).cuda()
        train_loss_meter = AverageMeter()
        if train_mode==True:
            model.train()
            #schedule_lr(epoch,OPTIMIZER)
            for inputs, inlabels, outlabels, inlabel_lens in tqdm(iter(train_loader)):
                # inputs:b*c*t*h*w btchw
                # inlabels:b*seq_len
                # inlabel_lens:b*1

                inputs=inputs.to(DEVICE)
                inlabels=inlabels.to(DEVICE)
                outlabels=outlabels.to(DEVICE)#b*seq_len
                '''tgt_key_padding_mask:
                the last several positions is filled with true.
                [ False, False, False,.....,  True,  True]
                '''
                tgt_key_padding_mask=[[False]*label_len.item()+[True]*(DECODER_MAX_SEQ_LEN-label_len.item()) for label_len in inlabel_lens]
                tgt_key_padding_mask=torch.BoolTensor(tgt_key_padding_mask).cuda()

                output=model(inputs, inlabels,inlabel_lens,tgt_mask
                             # ,tgt_key_padding_mask=tgt_key_padding_mask
                             )#b*seq_len*vocabulary_size
                output=output.view(BATCH_SIZE*DECODER_MAX_SEQ_LEN,VOCABULARY_SIZE)
                #print(output[0].argmax(dim=1))
                #output=torch.transpose(output,1,2)#b*vocabulary_size*seq_len
                outlabels=outlabels.view(BATCH_SIZE*DECODER_MAX_SEQ_LEN)
                print (''.join([chr(96+i)if i!=27 else ' 'for i in outlabels.tolist()]))
                print ('-------')
                pre=output.argmax(dim=1)
                print (''.join([chr(96 + i) if i != 27 else ' ' for i in pre.tolist()]))
                loss=LOSS(output,outlabels)
                train_loss_meter.update(loss.data.item(),inputs.size(0))
                print("loss=%.3f\n" % loss.data.item())

                # for params in OPTIMIZER.param_groups:
                #     params['lr'] = d_model ** (-0.5) * \
                #                    min((factor*step)**(-0.5),(factor*step)*(warmupsteps**(-1.5)))
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
            if seen=='seen':
                if train_loss_meter.avg < 0.025:
                    val_mode = True
                else:
                    val_mode = False
            elif seen=='unseen':
                if train_loss_meter.avg < 0.04:
                    test_mode = True
                else:
                    test_mode = False

            print('avg_loss=%.3f' % train_loss_meter.avg)

        valwermeter = AverageMeter()
        if val_mode==True:
            model.eval()

            for inputs, labels in tqdm(iter(val_loader)):
                # inputs:b*c*t*h*w labels:b strings
                inputs = inputs.to(DEVICE)

                #last_predict = torch.LongTensor([28] + (MAX_SEQ_LEN-1) * [0]).squeeze(0).expand(BATCH_SIZE, -1).cuda()  # b*seq_len
                last_predict = torch.LongTensor([28] + (DECODER_MAX_SEQ_LEN-1) * [0]).squeeze(0).expand(BATCH_SIZE, -1).cuda() # b*seq_len
                # print(last_predict)
                predicted_labels = torch.zeros(BATCH_SIZE, DECODER_MAX_SEQ_LEN,dtype=torch.int8)
                for i in range(DECODER_MAX_SEQ_LEN):
                    #print(last_predict)
                    # output=transformer(inputs,last_predict,input_len,infer=i)#b*seq_len*voc_size
                    tgt_key_padding_mask = [False] * (i+1) + [True] * (DECODER_MAX_SEQ_LEN - i-1)
                    tgt_key_padding_mask = torch.BoolTensor(tgt_key_padding_mask)
                    tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(0).expand(inputs.size(0), -1).cuda()
                    label_len_val=torch.LongTensor(inputs.size(0)*[i+1])
                    output = model(inputs, last_predict,label_len_val,tgt_mask
                                   ,tgt_key_padding_mask
                                   )  # b*seq_len*voc_size
                    # print(output)
                    output = output[:, i]  # b*voc_size
                    output = torch.argmax(output, dim=1)  # b

                    # print(output.size())
                    # print(last_predict[:,i+1].size())
                    if i < (DECODER_MAX_SEQ_LEN-1):
                        temp_last_predict=last_predict.clone()
                        temp_last_predict[:, i + 1] = output
                        last_predict=temp_last_predict.cuda()
                    predicted_labels[:, i] = output
                    #print (predicted_labels)
                predicted_labels = predicted_labels.tolist()  # b*seq_len
                error, gt = calculateWer(labels, predicted_labels, spell)
                valwermeter.update(float(error) / gt, inputs.size(0))
            print('val_avg_wer=%.3f' % valwermeter.avg)
            if val_once==True:
                break

        testwermeter=AverageMeter()
        if test_mode == True:
            model.eval()
            for inputs, labels in tqdm(iter(test_loader)):
                # inputs:b*c*t*h*w labels:b strings
                inputs=inputs.to(DEVICE)
                last_predict=torch.LongTensor([28]+(DECODER_MAX_SEQ_LEN-1)*[0]).squeeze(0).expand(BATCH_SIZE,-1).cuda()#b*seq_len
                #print(last_predict)
                predicted_labels=torch.zeros(BATCH_SIZE,DECODER_MAX_SEQ_LEN,dtype=torch.int8)
                for i in range(DECODER_MAX_SEQ_LEN):
                    tgt_key_padding_mask = [False] * (i + 1) + [True] * (DECODER_MAX_SEQ_LEN - i - 1)
                    tgt_key_padding_mask = torch.BoolTensor(tgt_key_padding_mask)
                    tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(0).expand(inputs.size(0), -1).cuda()
                    label_len_val = torch.LongTensor(inputs.size(0) * [i + 1])
                    output = model(inputs, last_predict, label_len_val, tgt_mask
                                   , tgt_key_padding_mask
                                   )  # b*seq_len*voc_size
                    # print(output)
                    output = output[:, i]  # b*voc_size
                    output = torch.argmax(output, dim=1)  # b

                    # print(output.size())
                    # print(last_predict[:,i+1].size())
                    if i < (DECODER_MAX_SEQ_LEN - 1):
                        temp_last_predict = last_predict.clone()
                        temp_last_predict[:, i + 1] = output
                        last_predict = temp_last_predict.cuda()
                    predicted_labels[:, i] = output
                    # print (predicted_labels)
                predicted_labels = predicted_labels.tolist()  # b*seq_len
                error, gt = calculateWer(labels, predicted_labels, spell)
                testwermeter.update(float(error) / gt, inputs.size(0))
            print('test_avg_wer=%.3f' % testwermeter.avg)
            if test_once==True:
                break

        #log_file.write('{:15}{:15}{:15}\n'.format('epoch','val_wer','train_loss'))
        #log_file.write('{:15}{:15.3f}{:15.3f}\n'.format(epoch, valwermeter.avg, train_loss_meter.avg))
        model_name = '%s_Time%s_epoch%d'%(remark,get_time(),epoch+init_epoch)
        if train_mode==True:
            model_name+='_loss%.3f'%(train_loss_meter.avg)
        if test_mode==True:
            model_name+='_test%.3f'%(testwermeter.avg)
        if val_mode==True:
            model_name+='_val%.3f'%(valwermeter.avg)
        if save_ckpt==True:
            torch.save(model.state_dict(),os.path.join('trancheckpoint',model_name))
