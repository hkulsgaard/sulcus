import torch
from torch import nn
import myModule
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

class MyNet(myModule.myModule):
    def __init__(self, pretrained, h_size, criterion, dropout=0, n_classes=None, freeze = 'False'):
        super(MyNet, self).__init__(criterion=criterion)
        self.aucs = {"train":[],"val":[]}
        self.accuracies = {"train":[],"val":[]}
        self.baccs = {"train":[],"val":[]}
        self.f1s = {"train":[],"val":[]}
        self.best_f1 = 0
        self.best_acc = 0
        self.best_bacc = 0
        self.y_hats = {"train":[],"val":[]}
        self.probs_y_hats = {"train":[],"val":[]}
        self.labels = {"train":[],"val":[]}
        self.h_size = h_size
        self.freeze = freeze

        self.pretrained = pretrained

        '''
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.h_size, n_classes)

        if n_classes<=2:
            self.s = nn.Sigmoid()
        else:
            self.s = nn.Softmax()


        self.mynet = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128,n_classes),
            nn.Sigmoid()
        )
        '''
        
        self.mynet = nn.Sequential(
            nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
            #nn.Sigmoid()
        )
        
        

    def forward(self, x):
        x = self.pretrained(x)
        #x = x.view(-1, self.num_flat_features(x)) #plancho las features
        #x = self.flatten(x)
        #x = self.dropout1(x)
        #x = self.fc1(x)
        #x = self.s(x)

        # Para recorrer todas las layers en "nn.Sequential"
        for layer in self.mynet:
            x = layer(x)
            #print(x.size())
        
        x = torch.sigmoid_(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def plot_epoch_metrics(self, phase, epoch, lr, preview_img, plotter):
        plotter.plot_value(plot_name = 'Loss', split_name=phase, x=epoch, y=self.losses[phase][-1])
        #plotter.plot_value(plot_name = 'Accuracy', split_name=phase, x=epoch, y=self.accuracies[phase][-1])
        plotter.plot_value(plot_name = 'Balanced Accuracy', split_name=phase, x=epoch, y=self.baccs[phase][-1])
        plotter.plot_value(plot_name = 'F1 Score', split_name=phase, x=epoch, y=self.f1s[phase][-1])
        plotter.plot_value(plot_name = 'Learning Rate(x10000)', split_name=phase, x=epoch, y=lr*10000)

    def calculate_epoch_metrics(self, phase):
        #print("labels: ", self.labels[phase])
        #print("probs_y_hats: ", self.probs_y_hats[phase])
        #print("y_hats: ", self.y_hats[phase])


        #all this block is for age
        # Compute confusion matrix
        cm = confusion_matrix(self.labels[phase], self.y_hats[phase])
        print(cm)      
        
        accuracy = accuracy_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]Accuracy {} : {:.3f}'.format(phase, accuracy))
        self.accuracies[phase].append(accuracy)

        bal_acc = balanced_accuracy_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]Bal Acc {} : {:.3f}'.format(phase, bal_acc))
        self.baccs[phase].append(bal_acc)
        
        f1 = f1_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]F1 Score {} : {:.3f}'.format(phase, f1))
        self.f1s[phase].append(f1)

        if not phase == 'test':
            #age -> reinicio esto porque sino me muestra demasiado
            self.y_hats = {"train":[],"val":[]}
            self.labels = {"train":[],"val":[]}

        #auc = roc_auc_score(y_true=self.labels[phase], y_score=self.y_hats[phase], multi_class='ovr')
        #self.aucs[phase].append(auc)
        #print('[INFO]AUC',phase+':', auc)

    def calculate_extended_metrics(self, phase):
        tn, fp, fn, tp = confusion_matrix(self.labels[phase], self.y_hats[phase]).ravel()
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        print('[INFO]Sensitivity {} : {:.3f}'.format(phase, sensitivity))
        print('[INFO]Specificity {} : {:.3f}'.format(phase, specificity))

        precision = precision_score(y_true = self.labels[phase], y_pred = self.y_hats[phase]) 
        print('[INFO]Precision {} : {:.3f}'.format(phase, precision))

    def calculate_loss(self,b_volumes,b_labels,phase):
        #b_labels = b_labels.type(torch.cuda.FloatTensor) #BCE
        b_labels = b_labels.to(self.device)#ce

        probs = self(b_volumes)
        #probs.to(self.device)

        # Selects the class with greater probability -> predicted class
        #y_hat = probs.round()
        #y_hat = torch.argmax(probs,1)
        probs_loss, y_hat = torch.max(probs,1)

        # Por cada imagen se le asigna un 1 en la columna que corresponda 
        # (segun su clase)
        # El resto de las columnas tienen 0
        target = torch.zeros(probs.shape).to(self.device)
        for i in range(target.shape[0]):
            target[i,b_labels[i].long()] = 1
        
        self.labels[phase] = self.labels[phase] + b_labels.tolist() #age
        self.y_hats[phase] = self.y_hats[phase] + y_hat.tolist() #age
        self.probs_y_hats[phase] = self.probs_y_hats[phase] + probs.tolist() #age
        
        loss = self.criterion(probs, target) #BCE
        #loss = self.criterion(probs, b_labels) #CE

        return loss
    

    def build_checkpoint(self,epoch,epoch_loss):
        checkpoint = super(MyNet, self).build_checkpoint(epoch,epoch_loss)
        checkpoint.update({'accuracies':self.accuracies, 
                           'baccs':self.baccs,
                           'f1s':self.f1s})
        
        return checkpoint

    def load_from_checkpoint(self, path=None, verbose=True):
        try:
            checkpoint = super().load_from_checkpoint(path, verbose=False)

            self.accuracies = checkpoint['accuracies']
            self.baccs = checkpoint['baccs']
            self.f1s = checkpoint['f1s']
            self.losses = checkpoint['losses']
        
            print('[INFO]Checkpoint restored-> Last_epoch:{} | Last saved epoch:{} | Best loss: {:.3f})'.format(self.last_epoch,
                                                                                                            self.last_saved_epoch,
                                                                                                            self.best_loss))

        except Exception as e:
            print('[INFO]Cannot restore checkpoint')
            print('[ERROR]',e)
            print()

    
    def load_from_best_model(self, path=None, verbose=False):
        try:
            best_model = super().load_from_best_model(path)

            try:
                self.losses = best_model['losses']
                self.accuracies = best_model['accuracies']
                self.baccs = best_model['baccs']
                self.f1s = best_model['f1s']
                print('[INFO]Best model restored-> Epoch:{} | Loss: {:.3f} | F1: {:.3f} | BAcc: {:.3f})'.format(
                    self.last_epoch,
                    self.losses['val'][-1],
                    self.f1s['val'][-1],
                    self.baccs['val'][-1]))
            except:
                print('[INFO]Epoch {} | Loss:{:.3f}'.format(self.last_epoch, self.losses['val'][-1]))

        except Exception as e:
            print('[INFO]Cannot restore best model')
            print('[ERROR]',e)
            print()
        
        return best_model
    

    def predict(self,input):
        return self(input)
    
    def get_parameters(self, lr, factor=1):
        # Returns the parameters splited in the pretrained network and the last layer of classification
        # This parameters are setted to be introduced directly to the optimizer

        pretrained_params, mynet_params = [], []
        for pname, p in self.named_parameters():
            if pname.find('pretrained') >= 0:
                pretrained_params.append(p)
            elif pname.find('mynet') >= 0:
                mynet_params.append(p)
        
        parameters = [{ 'params': pretrained_params, 'lr': lr },
                      { 'params': mynet_params, 'lr': lr*factor }]

        return parameters
    

    def freeze_pretrained(self):
        if self.freeze:
            print('[INFO]Freezing encoder')
          
            for param in self.pretrained.parameters():
                param.requires_grad = False
        '''
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
    
    def save_best_model(self,epoch,epoch_loss,model_dir):
        print('[INFO]Best F1: {:.3f}'.format(self.best_f1))
        print('[INFO]Best Acc: {:.3f}'.format(self.best_acc))
        if (self.f1s['val'][-1] >= self.best_f1):    
            #self.best_loss = epoch_loss
            self.last_saved_epoch = epoch
            self.best_f1 = self.f1s['val'][-1]
            self.best_acc = self.accuracies['val'][-1]
            self.best_bacc = self.baccs['val'][-1]
            
            best_model = self.build_best_model(epoch,epoch_loss)
            torch.save(best_model, model_dir + '/best_model.pt')
            print('[INFO]Best model saved (f1)')

        else:
            print('[INFO]Epoch #{} worst f1 than epoch #{}'.format(epoch,self.last_saved_epoch))

        if (self.baccs['val'][-1] >= self.best_bacc):
            self.best_bacc = self.baccs['val'][-1]
            best_model = self.build_best_model(epoch,epoch_loss)
            torch.save(best_model, model_dir + '/best_model_bacc.pt')
            print('[INFO]Best model saved (balanced accuracies)')

        if (epoch_loss < self.best_loss):    
            self.best_loss = epoch_loss
            best_model = self.build_best_model(epoch,epoch_loss)
            torch.save(best_model, model_dir + '/best_model_loss.pt')
            print('[INFO]Best model saved (loss)')

    def build_best_model(self, epoch, epoch_loss):
        
        best_model = super().build_best_model(epoch,epoch_loss)
        best_model['accuracies'] = self.accuracies
        best_model['baccs'] = self.baccs
        best_model['f1s'] = self.f1s

        return best_model