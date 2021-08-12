import pandas as pd
import torch
from torch import nn as nn
import myModule

# Encoder

# TODO: desharcodear la dimensión inicial

#####################
###### Encoder ######
#####################

class Encoder(nn.Module):
    def __init__(self, csv_file, kernel_size_maxpool=2, prob_dropout=0):
        super(Encoder, self).__init__()        
        conf_file = pd.read_csv(csv_file)
        self.blocks = nn.ModuleList()
        self.max_pool = nn.MaxPool3d(kernel_size=kernel_size_maxpool, stride=2, padding=0)

        for i,j in conf_file.iterrows():
            self.blocks.append(Block_encoder(conf_file.iloc[i,0],conf_file.iloc[i,1],conf_file.iloc[i,2],conf_file.iloc[i,3],prob_dropout=prob_dropout))
          
        
    def forward(self,input):
        h=input
        for i in range(len(self.blocks)-1): 
            h = self.blocks[i](h)
            h = self.max_pool(h)   
        h = self.blocks[len(self.blocks)-1](h)
        return h



###### Bloque encoder ######    

class Block_encoder(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, kernel_size_conv1=3,kernel_size_conv2=3, prob_dropout=0):
        super(Block_encoder,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels_1, out_channels=out_channels_1,kernel_size=kernel_size_conv1,padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        self.conv2 = nn.Conv3d(in_channels=in_channels_2, out_channels=out_channels_2,kernel_size=kernel_size_conv2,padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout3d(p=prob_dropout) 
        
 

    def forward(self, input):
        h = input
        h = self.conv1(h)
        h = self.bn1(h)
        h = nn.functional.relu(h) 
        h = self.conv2(h)
        h = self.bn2(h)
        h = nn.functional.relu(h)
        h = self.dropout(h)
          
        return h


#####################
###### Decoder ######
#####################    
    
    
class Decoder(nn.Module):
    def __init__(self,csv_file,prob_dropout):
        super(Decoder, self).__init__()
        conf_file = pd.read_csv(csv_file)
        self.blocks = nn.ModuleList()
        rows = conf_file.shape[0]
        
        for i in range(0,rows-1):
            
            self.blocks.append(Block_decoder(conf_file.iloc[i,0], conf_file.iloc[i,1], conf_file.iloc[i,2], conf_file.iloc[i,3], prob_dropout=prob_dropout))
 
        self.blocks.append(Block_decoder_sigmoid(conf_file.iloc[rows-1,0], conf_file.iloc[rows-1,1], conf_file.iloc[rows-1,2], conf_file.iloc[rows-1,3], prob_dropout=prob_dropout))
        
    
    def forward(self, input):
        x_hat = input  
        
        for i in range(len(self.blocks)-1):
            x_hat = nn.functional.interpolate(x_hat,scale_factor=(2,2,2),mode='trilinear')
            x_hat = self.blocks[i](x_hat)
        x_hat = self.blocks[len(self.blocks)-1](x_hat)
        return x_hat

###### Bloque decoder ######
        
class Block_decoder(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, kernel_size_conv1=3,kernel_size_conv2=3, prob_dropout=0):
        super(Block_decoder,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=kernel_size_conv1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=kernel_size_conv2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm3d(out_channels_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout3d(p=prob_dropout)
        

    def forward(self, input):
        x_hat = input
        x_hat = self.conv1(x_hat)
        x_hat = self.bn1(x_hat)
        x_hat = nn.functional.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.bn2(x_hat)
        x_hat = nn.functional.relu(x_hat)
        x_hat = self.dropout(x_hat)
       
        
        return x_hat

###### Bloque decoder final######
        
class Block_decoder_sigmoid(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, kernel_size_conv1=3,kernel_size_conv2=3, prob_dropout=0):
        super(Block_decoder_sigmoid,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=kernel_size_conv1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=kernel_size_conv2, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm3d(out_channels_2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout3d(p=prob_dropout)
        self.s = nn.Sigmoid()

    def forward(self, input):
        x_hat = input
        x_hat = self.conv1(x_hat)
        x_hat = self.bn1(x_hat)
        x_hat = nn.functional.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.dropout(x_hat)
        x_hat = self.s(x_hat)
       
        
        return x_hat

#########################
###### Autoencoder ######
#########################        
    
class Autoencoder(myModule.myModule):
    def __init__(self,conf_file_encoder,conf_file_decoder,criterion=None,prob_dropout=0,lr=None):
        super(Autoencoder,self).__init__(criterion,lr)
        self.encoder = Encoder(conf_file_encoder,prob_dropout=prob_dropout)
        self.decoder = Decoder(conf_file_decoder,prob_dropout=prob_dropout)

    def forward(self,input): 
        h = self.encoder(input) 
        x_hat = self.decoder(h) 
        
        return h, x_hat

    def calculate_loss(self,b_volumes,b_labels,phase):
        h, x_hat = self(b_volumes)
        x_hat = x_hat.type(torch.cuda.FloatTensor)
       
        return self.criterion(x_hat,b_volumes)

    def calculate_epoch_metrics(self,phase):
        #place holder
        return 0