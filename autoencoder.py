import pandas as pd
import torch
from torch import nn as nn
import myModule
import numpy as np

#####################
###### Encoder ######
#####################

class Encoder(nn.Module):
    def __init__(self, csv_file, kernel_size_maxpool=2, prob_dropout=0, h_size=None, filters = None, pre_h_shape=None):
        super(Encoder, self).__init__()
        self.h_size = h_size
        conf_file = pd.read_csv(csv_file)
        
        self.blocks = nn.ModuleList()
        self.max_pool = nn.MaxPool3d(kernel_size=kernel_size_maxpool, stride=2, padding=0)

        for i,j in conf_file.iterrows():
            self.blocks.append(Block_encoder(
                conf_file.iloc[i,0],conf_file.iloc[i,1],conf_file.iloc[i,2],conf_file.iloc[i,3],prob_dropout=prob_dropout))
            
        self.fc = nn.Linear(filters*pre_h_shape[0]*pre_h_shape[1]*pre_h_shape[2], self.h_size)#fc
          
        
    def forward(self, input):
        #cap: argument for captum (enables flattening of the embedding)

        h=input
        for i in range(len(self.blocks)-1):
            h = self.blocks[i](h)
            h = self.max_pool(h)

        h = self.blocks[len(self.blocks)-1](h)

        h = torch.flatten(h, 1)#fc
        h = self.fc(h)#fc

        # ONLY FOR CAPTUM (flattens the embedding)
        #if cap:
        if False:
            print('[INFO] original h out shape: {}'.format(h.shape)) #captum
            h = h.view(-1, h.size(1)) #captum
            print('[INFO] h out shape: {}'.format(h.shape)) #captum

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
    def __init__(self, csv_file, prob_dropout, h_size=None, pre_h_shape=None, filters=None):
        super(Decoder, self).__init__()
        self.h_size = h_size
        self.filters = filters
        self.pre_h_shape = pre_h_shape
        
        conf_file = pd.read_csv(csv_file)
        self.blocks = nn.ModuleList()
        rows = conf_file.shape[0]
        
        #for i,j in conf_file.iterrows():
        for i in range(rows-1):
            self.blocks.append(Block_decoder(conf_file.iloc[i,0], conf_file.iloc[i,1], conf_file.iloc[i,2], conf_file.iloc[i,3], prob_dropout=prob_dropout))

        i = rows-1
        self.blocks.append(Block_decoder_final(conf_file.iloc[i,0], conf_file.iloc[i,1], conf_file.iloc[i,2], conf_file.iloc[i,3]))
        
        self.fc = nn.Linear(self.h_size, filters*pre_h_shape[0]*pre_h_shape[1]*pre_h_shape[2])#fc

    def forward(self, input):
        x_hat = input
        
        x_hat = self.fc(x_hat)#fc
        x_hat = torch.reshape(x_hat, [-1,self.filters,self.pre_h_shape[0],self.pre_h_shape[1],self.pre_h_shape[2]])#fc
        
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
        
class Block_decoder_final(nn.Module):
    def __init__(self, in_channels_1, out_channels_1, in_channels_2, out_channels_2, kernel_size_conv1=3,kernel_size_conv2=1):
        super(Block_decoder_final,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels_1, out_channels=out_channels_1, kernel_size=kernel_size_conv1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels_1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv3d(in_channels=in_channels_2, out_channels=out_channels_2, kernel_size=kernel_size_conv2, padding=0)
                
        

    def forward(self, input):
        x_hat = input
        x_hat = self.conv1(x_hat)
        x_hat = self.bn1(x_hat)
        x_hat = nn.functional.relu(x_hat)
        x_hat = self.conv2(x_hat)
        
        return x_hat

#########################
###### Autoencoder ######
#########################        
    
class Autoencoder(myModule.myModule):
    def __init__(self,conf_file_encoder,conf_file_decoder,criterion=None,prob_dropout=0, lr=None, h_size=None, filters=None, pre_h_shape=None):
        super(Autoencoder,self).__init__(criterion,lr)
        self.encoder = Encoder(conf_file_encoder,prob_dropout=prob_dropout, h_size=h_size, filters=filters, pre_h_shape=pre_h_shape)
        self.decoder = Decoder(conf_file_decoder,prob_dropout=prob_dropout, h_size=h_size, filters=filters, pre_h_shape=pre_h_shape)

    def forward(self,input):
        #print('[INFO]input: {}'.format(input.shape)) 
        h = self.encoder(input)
        #print('[INFO]h: {}'.format(h.shape)) 
        x_hat = self.decoder(h)
        #print('[INFO]x_hat: {}'.format(x_hat.shape))  
        
        return h, x_hat

    def calculate_loss(self,b_volumes,b_labels,phase):
        h, x_hat = self(b_volumes)
        x_hat = x_hat.type(torch.cuda.FloatTensor)
       
        return self.criterion(x_hat,b_volumes)

    def calculate_epoch_metrics(self,phase):
        #place holder
        return 0
    
    def plot_epoch_metrics(self, phase, epoch, lr, preview_img, plotter):
        plotter.plot_value(plot_name = 'Loss', split_name=phase, x=epoch, y=self.losses[phase][-1])
        plotter.plot_value(plot_name = 'Learning Rate', split_name=phase, x=epoch, y=lr)

        #shows the preview img in visdom
        self.plot_preview_img(preview_img, plotter)


    def plot_preview_img(self, preview_img=None, plotter=None):
        
        if not preview_img==None:
            preview_img = preview_img.to(self.device)
            _ , x_hat = self.predict(preview_img)
            
            plotter.display_reconstruction(['img_reconstructed'],
                                           [x_hat[0,0,21,:,:].cpu().detach().numpy()],
                                           ['Reconstructed image'])
            '''
            plotter.display_reconstruction(['img_original','img_reconstructed'],
                                           [preview_img[0,0,7,:,:].cpu().numpy(), x_hat[0,0,7,:,:].cpu().detach().numpy()],
                                           ['Original image','Reconstructed image'])
            '''

    def predict(self,input):
        h, x_hat = self(input)
        
        return h, nn.functional.sigmoid(x_hat)