import pandas as pd
import torch
from torch import nn as nn
import autoencoder
import resnet
import os

#####################
##### DecoderRN #####
#####################    
    
class DecoderRN(resnet.ResNet):
    def __init__(self):
        super(resnet.ResNet, self).__init__()
        #self.inplanes = 64
        self.inplanes = 512
       
        # pasarlos a parámetros
        block = resnet.BasicBlock
        block_up = BasicBlockUp
        layers = [1,1,1,1]
        shortcut_type = 'B'

        '''
        self.conv_seg = nn.Sequential(
                            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
                            nn.Flatten(start_dim=1),
                            nn.Dropout(0.6),
                            nn.Linear(512 * block.expansion, num_seg_classes)
                            )
        '''                   
        
        self.layer1 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.layer2 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer3 = self._make_layer_up(block_up, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block,  64, layers[0], shortcut_type)

        #self.fc = nn.Linear(128, 512*32)#fc

        # último bloque
        self.conv1 = nn.ConvTranspose3d(64, 1, kernel_size=8, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer_up(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.ConvTranspose3d(self.inplanes, planes * block.expansion, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        
        #x = self.conv_seg(x)

        #x = self.fc(x)#fc
        #x = torch.reshape(x, [-1,512,2,4,4])#fc

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = nn.functional.interpolate(x,scale_factor=(2,2,2),mode='trilinear')
        x = self.conv1(x)

        
        #x = torch.sigmoid_(x) #comentado porque uso BCE con logits

        return x

class BasicBlockUp(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlockUp, self).__init__()
        self.conv1 = nn.ConvTranspose3d(inplanes, planes, kernel_size=4, dilation=dilation, stride=stride, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = resnet.conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = downsample
        
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        #print('>>UP')
        #print('input: {}'.format(x.shape))
        out = self.conv1(x)
        #print('conv1: {}'.format(out.shape))
        out = self.bn1(out)
        #print('bn1: {}'.format(out.shape))
        out = self.relu(out)
        #print('relu1: {}'.format(out.shape))
        out = self.conv2(out)
        #print('conv2: {}'.format(out.shape))
        out = self.bn2(out)
        #print('bn2: {}'.format(out.shape))

        if self.downsample is not None:
            residual = self.downsample(x)
            #print('downsample: {}'.format(residual.shape))


        out += residual
        out = self.relu(out)
        #print('relu2: {}\n'.format(out.shape))

        return out


#########################
##### AutoencoderRN #####
#########################        
    
class AutoencoderRN(autoencoder.Autoencoder):
    def __init__(self,pretrain_path=None,criterion=None,lr=None):
        super(autoencoder.Autoencoder,self).__init__(criterion,lr)
        self.encoder = self.init_encoder(pretrain_path)
        #self.encoder = self.init_encoder(pretrain_path, list(['conv_seg']))
        self.decoder = DecoderRN()
    
    def init_encoder(self, pretrain_path=None, new_layer_names=list()):
        # generate resnet10 model structure
        model = resnet.resnet10(
                sample_input_W=None,
                sample_input_H=None,
                sample_input_D=None,
                num_seg_classes=None)
                        
        #print('[INFO]init')
        #print(model.layer1[0].conv1.weight[0,0,0,:,:])

        #originalmente es un argumento (acá hardcodeado)
        gpu_id = 0
        
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
        model = model.to(self.device) 
        model = nn.DataParallel(model, device_ids=None)

        # load pretrain
        if pretrain_path == 'None':
            pretrain_path = None

        if not pretrain_path == None:
            print ('[INFO]Loading pretrained model {}'.format(pretrain_path))
            net_dict = model.state_dict()
            pretrain = torch.load(pretrain_path)
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
            
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

            # esto es originalmente un arg, acá está harcodeado (revisar setting.py del proyecto MedNet) 
            #new_layer_names = list(['conv_seg'])

            new_parameters = [] 
            for pname, p in model.named_parameters():
                for layer_name in new_layer_names:
                    if pname.find(layer_name) >= 0:
                        new_parameters.append(p)
                        break

            new_parameters_id = list(map(id, new_parameters))
            base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
            parameters = {'base_parameters': base_parameters, 
                        'new_parameters': new_parameters}

            #return model, parameters

        #return model, model.parameters()
        return model
    
    def predict(self,input):
        h, x_hat = self(input)
        
        return h, nn.functional.sigmoid(x_hat)