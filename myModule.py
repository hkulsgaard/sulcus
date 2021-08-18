import torch
from torch import nn
import os
import time
import matplotlib.pyplot as plt

class myModule(nn.Module):

    def __init__(self,criterion=None,lr=None):
        super(myModule,self).__init__()
        self.criterion = criterion
        self.last_epoch = 0
        self.last_saved_epoch = 0
        self.best_loss = 100
        self.losses = {"train":[],"val":[]}
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def set_optimizer(self,optimizer):
        self.optimizer = optimizer

    def set_scheduler(self,scheduler):
        self.scheduler = scheduler

    def load_from_checkpoint(self, checkpoint=None):
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.last_epoch = checkpoint['last_epoch']
            self.last_saved_epoch = checkpoint['last_saved_epoch']
            self.losses = checkpoint['losses']
            self.best_loss = checkpoint['best_loss']
            #lr = checkpoint['lr']
        
            print('[INFO]Checkpoint restored-> Last_epoch:{} | Last saved epoch:{} | Best loss: {})'.format(self.last_epoch,self.last_saved_epoch,self.best_loss))

    def load_from_checkpoint_2(self, path=None):
        try:
            checkpoint = torch.load(path[0])

            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.last_epoch = checkpoint['last_epoch']
            self.last_saved_epoch = checkpoint['last_saved_epoch']
            self.losses = checkpoint['losses']
            self.best_loss = checkpoint['best_loss']
            #lr = checkpoint['lr']
        
            print('[INFO]Checkpoint restored-> Last_epoch:{} | Last saved epoch:{} | Best loss: {})'.format(self.last_epoch,self.last_saved_epoch,self.best_loss))

        except:
            print('[INFO]Cannot restore checkpoint')

    def load_from_best_model(self, path):
        try:
            best_model = torch.load(path)

            self.load_state_dict(best_model['model_state_dict'])
            self.optimizer.load_state_dict(best_model['optimizer_state_dict'])
            self.scheduler.load_state_dict(best_model['scheduler_state_dict'])
            
            self.last_epoch = best_model['last_epoch']
            self.losses = best_model['losses']
            self.best_loss = best_model['best_loss']
        
        except:
            print('[INFO]Cannot load the model')

    def train_model(self, data_loaders, data_lengths, results_dir, n_epochs, verbose=True):
        
        start_train = time.time()
        for epoch in range(self.last_epoch+1, n_epochs+1):

            print('\n[INFO]Epoch #{} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(epoch))
            start = time.time()
            for phase in ['train','val']:
                print('[INFO]Phase: {} in progress...'.format(phase))
                
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0

                for batch_data in data_loaders[phase]:
                    b_images, b_labels = batch_data
                    b_images = b_images.to(self.device)
                    self.optimizer.zero_grad()

                    loss = self.calculate_loss(b_images,b_labels,phase)               
                    
                    if verbose:
                        print(loss.item())
                
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    running_loss += loss.item()
                
                epoch_loss = running_loss / data_lengths[phase]
                self.losses[phase].append(epoch_loss)
                print('[INFO]Epoch #{} loss: {}'.format(epoch,epoch_loss))                                       
                self.scheduler.step(epoch_loss)

                #calculate metrics for training and validation
                self.calculate_epoch_metrics(phase)

            #save best model if we got better loss
            self.save_best_model(epoch,epoch_loss,results_dir)
                        
            #save checkpoint for safety
            self.save_checkpoint(epoch,epoch_loss,results_dir)
            
            #remove old checkpoint
            self.delete_checkpoint(epoch=epoch-1,results_dir=results_dir)
            
            #save the losses plot
            self.build_plot(data=self.losses,data_name='Loss',results_dir=results_dir,verbose=False)

            elapsed = time.time() - start
            print('[TIME]Elapsed: {} sec ({:.1f} min)'.format(int(elapsed),float(elapsed/60)))

        #play_finish_sound()
        elapsed_total = time.time() - start_train
        print('[INFO]Job done -> Total time: {} sec ({:.1f} min)'.format(int(elapsed_total),float(elapsed_total/60)))

    def build_plot(self,data,data_name,ylim=None,results_dir=None ,verbose=True):
        plt.plot(data['train'], color='#0b97e3')
        plt.plot(data['val'], color='#e3bb0b')
        plt.legend(['Training '+data_name,'Validation '+data_name])
        plt.xlabel('Epoch')
        plt.ylabel(data_name)

        if ylim is not None:
            plt.ylim(0,ylim)
        if results_dir is not None:
            plt.savefig(results_dir + '/'+data_name+'.png')
        if verbose:
            plt.show()
        
    def calculate_epoch_metrics(self,phase):
        #place holder
        return 0

    def save_best_model(self,epoch,epoch_loss,results_dir):
        print('[INFO]Best loss: {}'.format(self.best_loss))
        if (epoch_loss < self.best_loss):
            self.best_loss = epoch_loss
            self.last_saved_epoch = epoch
            best_model = self.build_best_model(epoch,epoch_loss)
            torch.save(best_model, results_dir + '/best_model.pt')
            print('[INFO]Best model saved')
        else:
            print('[INFO]Epoch #{} worst loss than epoch #{}'.format(epoch,self.last_saved_epoch))

    def save_checkpoint(self,epoch,epoch_loss,results_dir):
        checkpoint = self.build_checkpoint(epoch,epoch_loss)
        torch.save(checkpoint, results_dir + '/checkpoint_epoch_{}.pt'.format(epoch))
        print('[INFO]Checkpoing epoch #{} saved'.format(epoch))

    def build_checkpoint(self,epoch,epoch_loss):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learning_rate': self.lr,
            
            'last_epoch': epoch,
            'last_saved_epoch' : self.last_saved_epoch,
            'losses' : self.losses,
            'best_loss' : self.best_loss,
            'epoch_loss': epoch_loss,
            }
        return checkpoint
    
    def build_best_model(self,epoch,epoch_loss):
        best_model = {
            #'model_state_dict': self.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            #'epoch': epoch,
            #'loss': epoch_loss,
            #'scheduler_state_dict': self.scheduler.state_dict(),
            #'learning_rate': self.lr,

            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learning_rate': self.lr,

            'last_epoch': epoch,
            'last_saved_epoch' : self.last_saved_epoch,
            'best_loss': epoch_loss,
            'losses' : self.losses,
            }
        return best_model

    def delete_checkpoint(self,epoch,results_dir):
        try: 
            os.remove(results_dir + '/checkpoint_epoch_{}.pt'.format(epoch))
            print('[INFO]Checkpoing epoch #{} deleted'.format(epoch))  
        except:
            print('[WARNING]No checkpoint deleted')

