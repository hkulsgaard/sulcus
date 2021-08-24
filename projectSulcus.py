import torch
import matplotlib.pyplot as plt
from torch import nn as nn
from torch import optim as optim
import autoencoder
import myNet
import glob
import myProject

class projectSulcus(myProject.myProject):

    def __init__(self):
        super(projectSulcus,self).__init__()

    def run(self, n_epochs, lr, batch_size, dim, h_size, patience, factor_patience, freeze,
            dropout, csv_train_path, csv_validation_path, config_encoder, config_decoder, 
            ae_path, parches_dir, results_dir, crop_values=None, verbose=False):

        # create results directory
        self.make_results_dir(results_dir)

        # Data loading
        data_loaders, data_lengths = self.load_dataset(csv_train_path=csv_train_path ,\
                                                csv_validation_path=csv_validation_path,\
                                                parches_dir=parches_dir, batch_size=batch_size, dim=dim,\
                                                transform=self.get_crop_transformation(dim,crop_values))

        # Create empty autoencoder
        criterion = nn.BCELoss()
        pretrained_model=autoencoder.Autoencoder(conf_file_encoder=config_encoder, conf_file_decoder=config_decoder,prob_dropout=dropout,criterion=criterion)
        pretrained_model = pretrained_model.cuda()

        # Take encoder layers
        encoder = autoencoder.Encoder(config_encoder,2,0)
        encoder = encoder.cuda()

        # Optimizer 
        optimizer = optim.Adam(list(pretrained_model.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True, patience=patience, factor=factor_patience)
        pretrained_model.set_optimizer(optimizer)
        pretrained_model.set_scheduler(scheduler)

        pretrained_model.load_from_best_model(ae_path)

        pretrained_dict = pretrained_model.state_dict()
        encoder_dict = encoder.state_dict()

        # Overwrite Encoder
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict) 
        encoder.load_state_dict(encoder_dict)

        # Adding a fc layer
        model = myNet.MyNet(pretrained=encoder, h_size=h_size, criterion=nn.BCELoss(),dropout=dropout,freeze=freeze)
        model = model.cuda()

        model_optimizer = optim.Adam(list(model.parameters()), lr=lr)
        model_scheduler = optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min',verbose=True, patience=patience, factor=factor_patience)
        model.set_optimizer(model_optimizer)
        model.set_scheduler(model_scheduler)

        #pppath = glob.glob(results_dir + "/checkpoint_epoch*.pt")
        model.load_from_checkpoint_2(glob.glob(results_dir + "/checkpoint_epoch*.pt"))

        model.train_model(data_loaders=data_loaders,data_lengths=data_lengths,results_dir=results_dir,n_epochs=n_epochs,verbose=verbose)

        # Metrics plots
        plt.figure(1)
        model.build_plot(data=model.losses,data_name='Loss',ylim=3,verbose=True,results_dir=results_dir)
        plt.figure(2)
        model.build_plot(data=model.aucs,data_name='AUC',ylim=1.1,verbose=True,results_dir=results_dir)
        plt.figure(3)
        model.build_plot(data=model.accuracies,data_name='Accuracy',ylim=1.1,verbose=True,results_dir=results_dir)
        plt.show()
        

        print('\n[INFO]Job done!\n')