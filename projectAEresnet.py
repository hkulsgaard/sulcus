import projectAE
import autoencoderResnet
from torch import optim as optim
from torch import nn as nn

class projectAEresnet(projectAE.projectAE):

    def __init__(self):
        super(projectAEresnet,self).__init__()
        self.ae = None

    def init_ae(self, config):
        
        #autoencoder creation
        criterion = nn.BCEWithLogitsLoss()
        self.ae = autoencoderResnet.AutoencoderRN(criterion=criterion, pretrain_path=config.get('pretrain_path'), lr = config['lr'])
        self.ae = self.ae.cuda()
        print(self.ae)
        
        optimizer = optim.Adam(list(self.ae.parameters()), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min',
            verbose=True,
            patience=config['patience'],
            factor=config['factor'],
            threshold = config['threshold'])
        
        self.ae.set_optimizer(optimizer)
        self.ae.set_scheduler(scheduler)

    def show_weight_test(self, msg='resnet encoder weight sample'):
        print('[INFO]{}'.format(msg))
        print(self.ae.encoder.module.layer1[0].conv1.weight[0,0,0,:,:])