import torch
from torch import nn
import myModule
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class MyNet(myModule.myModule):
    def __init__(self, pretrained, h_size, criterion, dropout=0.5, n_classes=None, freeze = 'False'):
        super(MyNet, self).__init__(criterion=criterion)
        self.aucs = {"train":[],"val":[]}
        self.accuracies = {"train":[],"val":[]}
        self.y_hats = {"train":[],"val":[]}
        self.probs_y_hats = {"train":[],"val":[]}
        self.labels = {"train":[],"val":[]}
        self.h_size = h_size

        self.pretrained = pretrained
        
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.h_size, n_classes)

        if n_classes<=2:
            self.s = nn.Sigmoid()
        else:
            self.s = nn.Softmax()
        
        '''
        # Clasificadores de 8 clases (para age)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.h_size, 8),
            nn.Sigmoid())
        '''
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(self.h_size,128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        
        

    def forward(self, x):
        x = self.pretrained(x)
        #x = x.view(-1, self.num_flat_features(x)) #plancho las features
        #x = self.flatten(x)
        #x = self.dropout1(x)
        #x = self.fc1(x)
        #x = self.s(x)

        # Para recorrer todas las layers en "nn.Sequential"
        for layer in self.net:
            x = layer(x)
            #print(x.size())

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def calculate_epoch_metrics(self,phase):
        #print("labels: ", self.labels[phase])
        #print("probs_y_hats: ", self.probs_y_hats[phase])
        #print("y_hats: ", self.y_hats[phase])


        #all this block is for age
        # Compute confusion matrix
        cm = confusion_matrix(self.labels[phase], self.y_hats[phase])
        print(cm)
        # Show confusion matrix in a separate window
        #plt.matshow(cm)
        #plt.title('Confusion matrix')
        #plt.colorbar()
        #plt.ylabel('True label')
        #plt.xlabel('Predicted label')
        #plt.show()
        
        # age (hacerlo funcionar porque lo necesito)
        #auc = roc_auc_score(y_true=self.labels[phase], y_score=self.y_hats[phase], multi_class='ovr')
        #self.aucs[phase].append(auc)
        #print('[INFO]AUC',phase+':', auc)
        
        accuracy = accuracy_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]Accuracy {} : {:.3f}'.format(phase, accuracy))
        self.accuracies[phase].append(accuracy)

        #age -> reinicio esto porque sino me muestra demasiado
        self.y_hats = {"train":[],"val":[]}
        self.labels = {"train":[],"val":[]}


    def calculate_loss(self,b_volumes,b_labels,phase):
        b_labels = b_labels.type(torch.cuda.FloatTensor)
        #b_labels = b_labels.type(torch.cuda.LongTensor) #multiclass

        probs = self(b_volumes)
        probs = probs.view(-1)
        probs.to(self.device)

        # Selects the class with greater probability -> predicted class
        y_hat = probs.round()
        #y_hat = torch.argmax(probs,1)
        
        self.labels[phase] = self.labels[phase] + b_labels.tolist() #age
        self.y_hats[phase] = self.y_hats[phase] + y_hat.tolist() #age
        self.probs_y_hats[phase] = self.probs_y_hats[phase] + probs.tolist() #age

        #print('y_hat:',y_hat.dtype)
        #print('b_labels:',b_labels.dtype)
        loss = self.criterion(probs, b_labels) #multiclass

        return loss

    def build_checkpoint(self,epoch,epoch_loss):
        checkpoint = super(MyNet, self).build_checkpoint(epoch,epoch_loss)
        checkpoint.update({'aucs':self.aucs, 'accuracies':self.accuracies})
        return checkpoint

    def load_from_checkpoint(self, path=None, verbose=True):
        try:
            checkpoint = super().load_from_checkpoint(path, verbose=False)

            self.aucs = checkpoint['aucs']
            self.accuracies = checkpoint['accuracies']
        
            print('[INFO]Checkpoint restored-> Last_epoch:{} | Last saved epoch:{} | Best loss: {:.3f})'.format(self.last_epoch,
                                                                                                            self.last_saved_epoch,
                                                                                                            self.best_loss))

        except Exception as e:
            print('[INFO]Cannot restore checkpoint')
            print('[ERROR]',e)
            print()

    
    '''
    poner esto en el inicializador
    #corregir porque ahora en vez de tener el encoder tengo todo el AE como pretrained
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
    '''