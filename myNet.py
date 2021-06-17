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
        print('AUC', auc)
        
        accuracy = accuracy_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('ACC', accuracy)
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

