import numpy as np
import torch

from torch.nn import functional as F1
from torch.nn import Transformer,BatchNorm3d,AdaptiveMaxPool2d,AdaptiveAvgPool2d,BatchNorm1d,LayerNorm,Embedding,Parameter,ModuleList,Softmax,Sequential,Linear,Dropout3d,Module,Conv3d,Conv2d,Conv1d,MaxPool3d,BatchNorm2d,ReLU,MaxPool2d

from stn_tps import STNHead,TPSSpatialTransformer
class BasicBlock3D(Module):
    '''
    A BasicBlock for Resnet which passes through 2 conv3d-bn-relu modules and
    adds residual unit on itself.
    channel:inplanes->outplanes.
    '''
    def __init__(self,inplanes,outplanes,downsample=None,stride=1):
        super(BasicBlock3D,self).__init__()
        self.conv1=Conv3d(inplanes,outplanes,kernel_size=(1,3,3),stride=stride,padding=(0,1,1),bias=True)
        self.bn1=BatchNorm3d(outplanes)
        self.relu=ReLU(inplace=True)
        self.conv2=Conv3d(outplanes,outplanes,kernel_size=(1,3,3),stride=stride,padding=(0,1,1),bias=True)
        self.bn2=BatchNorm3d(outplanes)
        self.downsample=downsample

    def forward(self, x):
        identity=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        if self.downsample is not None:#identity!identity!
            identity=self.downsample(identity)
        x+=identity
        x=self.relu(x)

        return x

class Convolutional_Feature_Extractor(Module):
    '''
    channel:3->512
    '''
    def __init__(self,d_model):
        super(Convolutional_Feature_Extractor,self).__init__()
        #(N,Cin,D,H,W)
        self.conv3d1=Conv3d(in_channels=3,out_channels=32,kernel_size=(5,5,5),padding=(2,2,2)
                           # ,stride=(1,2,2)
        )
        self.mp3d1 = MaxPool3d(kernel_size=(1, 2, 2))
        self.bn1=BatchNorm3d(32)
        self.basicblock1 = BasicBlock3D(inplanes=32,outplanes=32)

        self.conv3d2=Conv3d(in_channels=32,out_channels=64,kernel_size=(5,5,5),padding=(2,2,2)
                           # ,stride=(1,2,2)
                            )
        self.mp3d2 = MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))#(N,Cin,D,H',W')
        self.bn2 = BatchNorm3d(64)
        self.basicblock2 = BasicBlock3D(inplanes=64, outplanes=64)

        self.conv3d3 = Conv3d(in_channels=64, out_channels=96, kernel_size=(1, 5, 5), padding=(0, 2, 2)
                             # ,stride=(1,2,2)
                              )
        self.mp3d3 = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (N,Cin,D,H',W')
        self.bn3 = BatchNorm3d(96)
        self.basicblock3 = BasicBlock3D(inplanes=96, outplanes=96)

        self.gap=AdaptiveAvgPool2d((1,1))
        #self.linear=Linear(in_features=96*72,out_features=d_model)
        self.linear = Linear(in_features=96, out_features=d_model)
        self.bn=BatchNorm1d(d_model)

    def forward(self, x):
        #print (x)
        x=self.conv3d1(x)
        x=self.mp3d1(x)

        #x=self.bn1(x)
        x=self.basicblock1(x)

        x=self.conv3d2(x)
        x=self.mp3d2(x)
        #x = self.bn2(x)
        x = self.basicblock2(x)

        x = self.conv3d3(x)
        x = self.mp3d3(x)
        #x = self.bn3(x)
        x = self.basicblock3(x) #(N,Cin,D,6,12)
        x=torch.transpose(x,1,2) #(N,D,C,H',W')
        batch=x.size(0)
        time_step=x.size(1)
        x=x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)) #(N*T,C,H,W)          #(N,75,96*72)
        x=self.gap(x)#(N*T,C,1,1)
        x=x.squeeze(-1).squeeze(-1)#(N*T,96)
        x = self.linear(x)     #(N*T,512)                                                 # (N,75,512)
        x=x.view(batch,time_step,-1)#(N,T,512)
        x=torch.transpose(x,1,2)#(N,512,75)

        x=self.bn(x)
        x = torch.transpose(x, 1, 2)#(N,75,512)
        #assert x.size()==(BATCH_SIZE,TIME_STEP,d_model)
        return x

class PositionalEncoding(Module):
    '''
        positional encoding for the series.
        input:b*max_seq_len,each value ranges from [0,dictionary_size].
        e.g.[1,2,3,4,5,0,0,0,0,0]
        output:b*max_seq_len*d_mode;
    '''
    def __init__(self,max_seq_len,d_model):
        super(PositionalEncoding,self).__init__()#dic_size=max_seq_len+1
        self.posembedding=Embedding(max_seq_len+1,d_model)    #dictionary_size*d_model
        posencoding=np.array(
            [[t/np.power(10000,2.0*(d//2)/d_model) for d in range(d_model)]for t in range(1,1+max_seq_len)]
        )#seq_len*d_model
        posencoding[:,0::2]=np.sin(posencoding[:,0::2])
        posencoding[:,1::2]=np.cos(posencoding[:,1::2])
        posencoding=torch.Tensor(posencoding)
        posencoding=torch.cat((torch.zeros([1,d_model]),posencoding))
        self.posembedding.weight=Parameter(posencoding,requires_grad=False)

    def forward(self,x):#x:batch*seq_len
        output_pos=self.posembedding(x)
        return output_pos#b*t(l)*d

class Transformer_model(Module):
    def __init__(self,d_model=512,vocabulary_size=30,max_seq_len=75,decoder_max_seq_len=35,
                 nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu",stn_on=True,bs=16,ts=75):
        super(Transformer_model,self).__init__()
        self.stn_on=stn_on

        self.cfe=Convolutional_Feature_Extractor(d_model)#(n,t,e)
        self.positionEncoding=PositionalEncoding(max_seq_len,d_model)
        self.decodeEmbedding=Embedding(vocabulary_size,d_model)
        self.decoder_max_seq_len=decoder_max_seq_len
        self.transformer=Transformer(d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,activation=activation)
        self.linear=Linear(d_model,vocabulary_size)
        self.ln=LayerNorm(d_model)
        self.stn = STNHead(in_planes=3, num_ctrlpoints=20,
                           activation=None)
        self.tps = TPSSpatialTransformer(output_image_size=(50, 100),
                                         num_control_points=20,
                                         margins=tuple([0.05, 0.05]))
        self.bs=bs
        self.ts=ts
        #self.softmax=Softmax(dim=-1)!!!!!no!

    def forward(self,videos,labels,labels_len,tgt_mask,tgt_key_padding_mask=None):#videos:(N,Cin,D,H,W) label:(N,SEQ_LEN)
        #videos:bcthw

        if self.stn_on:
            print ('stn')
            videos=torch.transpose(videos,1,2)
            #print (videos.size())
            videos=videos.reshape(self.bs*self.ts,3,50,100)
            stn_input = F1.interpolate(videos, (32, 64), mode='bilinear', align_corners=True)#(bt)c3264
            stn_img_feat, ctrl_points = self.stn(stn_input)#(bt)n 2
            videos, _ = self.tps(videos, ctrl_points)
            videos=videos.view(self.bs,self.ts,3,50,100)
            videos=torch.transpose(videos,1,2)#bcthw
        src=self.cfe(videos)#(N,T,E)
        encoder_pos = torch.LongTensor([[t for t in range(1, 1 + src.size(1))] for b in range(src.size(0))]).cuda()#(N,T)
        encoding_pos = self.positionEncoding(encoder_pos)#(N,T,E)
        src+=encoding_pos#(N,T,E)
        src=torch.transpose(src,0,1)#(T,N,E)
        tgt=self.decodeEmbedding(labels)#(N,SEQ_LEN,E)
        decoder_pos = torch.LongTensor([[i for i in range(1, l.item() + 1)] + [0] * (self.decoder_max_seq_len - l.item()) for l in labels_len]).cuda()#(B,SEQ_LEN)
        decoding_pos=self.positionEncoding(decoder_pos)
        tgt+=decoding_pos#(N,SEQ_LEN,E)
        tgt=torch.transpose(tgt,0,1)#(SEQ_LEN,N,E)
        tranoutput=self.transformer(src,tgt,tgt_mask=tgt_mask
                                    ,tgt_key_padding_mask=tgt_key_padding_mask)#(seq_len,N,E)
        tranoutput=torch.transpose(tranoutput,0,1)#(N,seq_len,E)
        tranoutput=self.ln(tranoutput)
        tranoutput=self.linear(tranoutput)#(N,seq_len,voc_size)
        #output=self.softmax(tranoutput)#(N,seq_len,voc_size)
        output=tranoutput
        return output
