import torch
from torch import nn
import myModule
#from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

class MyNet(myModule.myModule):
    def __init__(self,pretrained,criterion,dropout=0.5,freeze = 'False'):
        super(MyNet, self).__init__(criterion=criterion)
        self.pretrained = pretrained
        self.aucs = {"train":[],"val":[]}
        self.accuracies = {"train":[],"val":[]}
        self.y_hats = {"train":[],"val":[]}
        self.probs_y_hats = {"train":[],"val":[]}
        self.labels = {"train":[],"val":[]}
        
        if (freeze == 'True'):
            print('freezing layers')
          
            for param in self.pretrained.parameters():
                param.requires_grad = False
            for param in self.pretrained.module.conv_seg.parameters():
                param.requires_grad = True
            for param in self.pretrained.module.layer4[0].parameters():
                param.requires_grad = True
            for param in self.pretrained.module.layer3[0].parameters():
                param.requires_grad = True
            for param in self.pretrained.module.layer2[0].parameters():
                param.requires_grad = True
            for param in self.pretrained.module.layer1[0].parameters():
                param.requires_grad = True
            for param in self.pretrained.module.layer1[0].parameters():
                param.requires_grad = True

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128*2*4*4, 1) 
        self.s = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained(x)
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))        
        x = self.dropout1(x)
        x = self.fc1(x)
        
        return self.s(x) #para BCEloss 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def calculate_epoch_metrics(self,phase):
        #print("labels: ", labels[phase])
        #print("probs_y_hats: ", self.probs_y_hats[phase])
        
        auc = roc_auc_score(y_true=self.labels[phase], y_score=self.y_hats[phase])
        self.aucs[phase].append(auc)
        print('[INFO]AUC',phase+':', auc)
        
        accuracy = accuracy_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]ACC',phase+':', accuracy)
        self.accuracies[phase].append(accuracy)

    def calculate_loss(self,b_volumes,b_labels,phase):
        y_hat = self(b_volumes)
        y_hat = y_hat.view(-1)
        #print('salida: ', y_hat)
        #print('etiquetas: ', b_labels)
        y_hat.to(self.device)
        b_labels = b_labels.type(torch.cuda.FloatTensor)

        #print("label masks: ", b_labels.tolist())
        self.labels[phase] = self.labels[phase] + b_labels.tolist()
        #print('labels val: ',labels_val)
        self.y_hats[phase] = self.y_hats[phase] + torch.round(y_hat).tolist()
        #print('out val: ',out_val)
        self.probs_y_hats[phase] = self.probs_y_hats[phase] + y_hat.tolist()

        return self.criterion(y_hat, b_labels)

    def build_checkpoint(self,epoch,epoch_loss):
        checkpoint = super(MyNet, self).build_checkpoint(epoch,epoch_loss)
        checkpoint.update({'aucs':self.aucs, 'accuracies':self.accuracies})
        return checkpoint

    def load_from_checkpoint(self, checkpoint=None):
        super(MyNet, self).load_from_checkpoint(checkpoint)
        self.aucs = checkpoint['aucs']
        self.accuracies = checkpoint['accuracies']

    def load_from_checkpoint_2(self, path=None):
        try:
            checkpoint = torch.load(path[0])

            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.last_epoch = checkpoint['last_epoch']
            self.last_saved_epoch = checkpoint['last_saved_epoch']
            self.best_loss = checkpoint['best_loss']

            self.losses = checkpoint['losses']
            self.aucs = checkpoint['aucs']
            self.accuracies = checkpoint['accuracies']
        
            print('[INFO]Checkpoint restored-> Last_epoch:{} | Last saved epoch:{} | Best loss: {})'.format(self.last_epoch,self.last_saved_epoch,self.best_loss))

        except:
            print('[INFO]Cannot restore checkpoint')

    