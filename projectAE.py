import myProject
import autoencoder
from torchsummary import summary
import glob
import torch
from torch import optim as optim
from torch import nn as nn

class projectAE(myProject.myProject):

    def __init__(self):
        super(projectAE,self).__init__()

    def run(self, n_epochs, latent_variable_dim, lr, batch_size, dim,\
        csv_train_path, csv_validation_path, parches_dir, results_dir,\
        csv_config_encoder, csv_config_decoder, crop_values=None):

        ###################### INITIAL SETUP ######################
        self.make_results_dir(results_dir)
        
        #dataset inicialization
        data_loaders, data_lengths = self.load_dataset(csv_train_path=csv_train_path ,\
                                                       csv_validation_path=csv_validation_path,\
                                        parches_dir=parches_dir, batch_size=batch_size, dim=dim,\
                                        transform=self.get_crop_transformation(dim,crop_values))


        #autoencoder creation
        ae = autoencoder.Autoencoder(conf_file_encoder=csv_config_encoder, conf_file_decoder=csv_config_decoder,\
                                                                    prob_dropout=0, criterion=nn.BCELoss(),lr=lr)
        ae = ae.cuda()
        print(ae)
        summary(ae, (1,64,64,32))
        optimizer = optim.Adam(list(ae.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True, patience=6, factor=0.5) 
        ae.set_optimizer(optimizer)
        ae.set_scheduler(scheduler)

        #checkpoint restoring
        checkpoint_path = glob.glob(results_dir + "/checkpoint_epoch*.pt")
        try:
            checkpoint = torch.load(checkpoint_path[0])
            ae.load_from_checkpoint(checkpoint)
        except:
            checkpoint = None

        #autoencoder training
        ae.train_model(data_loaders=data_loaders, data_lengths=data_lengths,results_dir=results_dir,n_epochs=n_epochs,verbose=False)

        print('\n[INFO]Job done!\n')