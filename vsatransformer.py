import torch.utils.data as data
import glob,os
import torchvision
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.nn import Transformer,BatchNorm3d,AdaptiveMaxPool2d,AdaptiveAvgPool2d,BatchNorm1d,LayerNorm,Embedding,Parameter,ModuleList,Softmax,Sequential,Linear,Dropout,Module,Conv3d,Conv2d,Conv1d,MaxPool3d,BatchNorm2d,ReLU,MaxPool2d
import torch.nn as nn
import torch.optim as optim
from utils import calculateWer,AverageMeter,get_time,Spell
from tqdm import tqdm
import random
from transformer_model import Transformer_model
from data_generator import Vsa_Datasetter_train,Vsa_Datasetter_test
# class Vsa_Datasetter_train(data.Dataset):
#     def __init__(self,video_dir,time_step,max_seq_len,rotate=False):
#         self.video_dir=video_dir
#         self.videos=glob.glob(os.path.join(self.video_dir,'*/*/*'))
#         self.video_num=len(self.videos)
#         self.time_step=time_step
#         self.train_transform=transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#              ]
#         )
#         self.max_seq_len=max_seq_len
#         self.rotate = rotate
#
#     def __getitem__(self, item):
#         path=self.videos[item]
#         pics=[]
#         angle = 0
#         if self.rotate == True:
#             angle = random.uniform(-10, 10)
#         for i in range(1,1+self.time_step):
#             pic = os.path.join(path, '%03d.jpg' % i)
#             pic = Image.open(pic).convert('RGB')#c*h*w
#             if self.rotate == True:
#                 pic = F.rotate(pic, angle)
#             pic = self.train_transform(pic)
#             pics.append(pic)
#         video=torch.stack(tuple(pics),dim=1)#c*t*h*w
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
#         num2word = {
#             '0': 'zero',
#             '1': 'one',
#             '2': 'two',
#             '3': 'three',
#             '4': 'four',
#             '5': 'five',
#             '6': 'six',
#             '7': 'seven',
#             '8': 'eight',
#             '9': 'nine'
#         }
#         str_num = ' '.join([num2word[digit] for digit in numbers])
#         label = [ord(digit) - ord('a') + 1 if digit != ' ' else 27 for digit in str_num]
#         inlabel = [28] + label + [29]
#         outlabel = label + [29]
#         inlabel_len = len(inlabel)
#         outlabel_len = len(outlabel)
#         inlabel += [0] * (self.max_seq_len - inlabel_len)
#         outlabel += [0] * (self.max_seq_len - outlabel_len)
#         return inlabel, outlabel, inlabel_len
#
# class Vsa_Datasetter_test(data.Dataset):
#     def __init__(self,video_dir,time_step):
#         self.video_dir=video_dir
#         self.videos=glob.glob(os.path.join(self.video_dir,'*/*/*'))
#         self.video_num=len(self.videos)
#         self.time_step=time_step
#         self.train_transform=transforms.Compose(
#             [
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
#              ]
#         )
#
#     def __getitem__(self, item):
#         path=self.videos[item]
#         pics=[]
#         for i in range(1,1+self.time_step):
#             pic=os.path.join(path,'%03d.jpg'%i)
#             pic=Image.open(pic).convert('RGB')
#             pic=self.train_transform(pic)
#             pics.append(pic)
#         video=torch.stack(tuple(pics),dim=1)
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
#         :param numbers:6315
#         :return: six three one five
#
#         pad:0
#         a-z:1-26
#         space:27
#         bos:28
#         eos:29
#         '''
#         num2word = {
#             '0': 'zero',
#             '1': 'one',
#             '2': 'two',
#             '3': 'three',
#             '4': 'four',
#             '5': 'five',
#             '6': 'six',
#             '7': 'seven',
#             '8': 'eight',
#             '9': 'nine'
#         }
#         str_num = ' '.join([num2word[digit] for digit in numbers])
#         return str_num
if __name__=='__main__':
    pid = os.getpid()
    f = open('trans_vsa.log', 'w')
    f.write(str(pid))
    f.close()
    spell = Spell('vsa.txt')
    BATCH_SIZE = 16#woc....increase it will cause CUDNN_STATUS_MAPPING_ERROR...WOCAO
    TIME_STEP = 50
    DECODER_MAX_SEQ_LEN = 25
    d_model=512
    VOCABULARY_SIZE=30
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    MAX_SEQ_LEN = 75

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    user="61"
    remark="{}original".format(user)
    #remark="original36"
    seen = 'unseen'

    ckpt='kernel551-seenNOSTN-866adam_Time2020-01-03-18-09_epoch59_loss0.013_val0.031'
    #ckpt='vsacheckpoint/64Stn(gridnostn)_Time2020-01-15-02-14_epoch35_loss0.042'

    #ckpt='trancheckpoint/tmp/kernel551-rotate-866adam_Time2019-12-15-02-13_epoch182_loss0.009_val0.028'
    #ckpt='kernel551-rotate-866adam_Time2019-12-15-02-13_epoch182_loss0.009_val0.028'
    #ckpt='vsacheckpoint/64_Time2020-01-02-22-48_epoch27_loss0.022_test0.090'

    stn_on=False
    #stn_on = True
    rotate=False

    train_mode = True
    val_mode = False
    val_once = False
    test_mode= False
    test_once=False
    save_ckpt=True
    epoch_init=0
    dataset_train='/ssddata/jinyu/lips_vsa{}/train'.format(user)
    dataset_test='/ssddata/jinyu/lips_vsa{}/test'.format(user)
    if seen=='seen':
        train_ds = Vsa_Datasetter_train('/ssddata/jinyu/lips_50100_keras/train', time_step=TIME_STEP, max_seq_len=DECODER_MAX_SEQ_LEN)
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, pin_memory=True)

        val_ds = Vsa_Datasetter_test('/ssddata/jinyu/lips_50100_keras/val', time_step=TIME_STEP)
        val_loader = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     drop_last=True, pin_memory=True)

        test_ds = Vsa_Datasetter_test('/ssddata/jinyu/lips_50100_keras/test', time_step=TIME_STEP)
        test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       drop_last=True, pin_memory=True)
    elif seen=='unseen':
        #all people except one train.One test.
        train_ds = Vsa_Datasetter_train(dataset_train, time_step=TIME_STEP,
                                        max_seq_len=DECODER_MAX_SEQ_LEN,rotate=rotate)
        train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       drop_last=True, pin_memory=True,num_workers=3)
        test_ds = Vsa_Datasetter_test(dataset_test, time_step=TIME_STEP)
        test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       drop_last=True, pin_memory=True,num_workers=3)
    #val_ds = Grid_Datasetter_test('/ssddata/jinyu/lips_grid/train', time_step=TIME_STEP, max_seq_len=MAX_SEQ_LEN)
    #val_loader = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
     #                            drop_last=True, pin_memory=True)
    model=Transformer_model(d_model,vocabulary_size=VOCABULARY_SIZE,max_seq_len=MAX_SEQ_LEN,
                         decoder_max_seq_len=DECODER_MAX_SEQ_LEN, nhead = nhead,
                         num_encoder_layers = num_encoder_layers,
                         num_decoder_layers = num_decoder_layers,dim_feedforward = dim_feedforward,
                         dropout = dropout,activation = activation
                         ,stn_on=stn_on,bs=BATCH_SIZE,ts=TIME_STEP)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(DEVICE)
    #model=model.cuda()
    #LOSS=nn.CrossEntropyLoss(ignore_index=0)
    LOSS=nn.CrossEntropyLoss()
    OPTIMIZER=optim.Adam(model.parameters(),lr=1e-4)
    #OPTIMIZER = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    #OPTIMIZER = optim.Adadelta(model.parameters(),lr=1)
    # for p in model.parameters():
    #     if p.dim() > 1 and p.requires_grad==True:
    #         nn.init.kaiming_normal(p)#add this cause 3.17 loss forever
    #         #nn.init.xavier_uniform(p)#add this cause 3.19 loss forever

    if not os.path.isdir('vsacheckpoint'):
        os.mkdir('vsacheckpoint')
    #log_file=open('log_{}'.format(remark),'w')
    #log_file.write('try adedlta by changing crossentrophy\n')
    for epoch in range(200):
        tgt_mask = torch.triu(torch.ones((DECODER_MAX_SEQ_LEN, DECODER_MAX_SEQ_LEN), dtype=torch.float) * float('-inf'),
                              diagonal=1).cuda()
        train_loss_meter = AverageMeter()
        if train_mode == True:
            model.train()

            #schedule_lr(epoch,OPTIMIZER)


            """
            tgt_mask:like this below,will be added to attention after Q(dotproduct)KT when decoding
            [[0., -inf, -inf, -inf, -inf],
            [0.,   0.,  -inf, -inf, -inf],
            [0.,   0.,   0.,  -inf, -inf],
            [0.,   0.,   0.,   0.,  -inf],
            [0.,   0.,   0.,   0.,   0.]]
            """

            for inputs, inlabels, outlabels, inlabel_lens in tqdm(iter(train_loader)):
                # inputs:b*c*t*h*w
                # inlabels:b*seq_len
                # inlabel_lens:b*1
                """test feed data,find no error
                print (inlabels[0])
                print (outlabels[0])
    
                pic=inputs[0]
                pic=torch.transpose(pic,0,1)#tchw
                pic=pic[0]#chw
                pic1= torchvision.transforms.ToPILImage()(pic)
                pic1.save('a_great_pic.jpeg')
                """
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
                              ,tgt_key_padding_mask=tgt_key_padding_mask
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
                #print("loss=%.3f" % loss.data.item())
                # for params in OPTIMIZER.param_groups:
                #     params['lr'] = d_model ** (-0.5) * \
                #                    min((factor*step)**(-0.5),(factor*step)*(warmupsteps**(-1.5)))
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
                if seen == 'seen':
                    if train_loss_meter.avg < 0.02:
                        val_mode = True
                    else:
                        val_mode = False
                elif seen == 'unseen':
                    if train_loss_meter.avg < 0.035:
                        test_mode = True
                    else:
                        test_mode = False

            print('avg_loss=%.3f' % train_loss_meter.avg)

        valwermeter = AverageMeter()
        if val_mode == True:
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
        model_name = '%s_Time%s_epoch%d'%(remark,get_time(),epoch+epoch_init)
        if train_mode==True:
            model_name+='_loss%.3f'%(train_loss_meter.avg)
        if test_mode==True:
            model_name+='_test%.3f'%(testwermeter.avg)
        if val_mode==True:
            model_name+='_val%.3f'%(valwermeter.avg)
        if save_ckpt==True:
            torch.save(model.state_dict(),os.path.join('vsacheckpoint',model_name))

    #log_file.close()