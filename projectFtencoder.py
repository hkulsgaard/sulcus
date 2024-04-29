import torch
from torch import nn as nn
from torch import optim as optim
import autoencoderResnet
import myNet
import myProject
import sulciDataset2
from torch.utils.data import DataLoader
from pandas import DataFrame
import os

class projectFtencoder(myProject.myProject):

    def __init__(self):        
        super(projectFtencoder,self).__init__()
        self.models = list()

    def run_predict(self, config):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Data loading
        dataset, dataLoader = self.load_dataset_test(config)
        
        # Loading the Ft-encoders
        for model_fname in config['models_fnames']:
            config['model_fname'] = model_fname
            model = self.load_model(config)
            model.eval()
            self.models.append(model)
            print('[INFO]Model loaded: "{}"'.format(model_fname))
        
        print()
        n_models = (len(self.models))

        # Initialize data dict
        data={'fnames':list(),
              'y_hats':list(),
              'certainty':list()}
        
        # Loop through the images
        for batch_id, (volume, label) in enumerate(dataLoader):
            y_hat_sum=0
            fname = dataset.get_fname(batch_id)
            volume = volume.to(device)
                        
            for model in self.models:
                with torch.no_grad():
                    probs = model(volume).cpu().detach()
                probs_loss, y_hat = torch.max(probs,1)
                y_hat_sum += y_hat.numpy().item()
            
            if y_hat_sum > (n_models/2):
                y_hat_majority = 1
                y_hat_label = 'Diagonal Sulcus detected'
            else:
                y_hat_majority = 0
                y_hat_label = 'Diagonal Sulcus NOT detected'

            certainty = round(abs((y_hat_sum*2-n_models)/n_models),1)

            # Store data about predictions
            data['fnames'].append(fname)
            data['y_hats'].append(y_hat_majority)
            data['certainty'].append(certainty)

            # Terminal output for predictions
            print('[INFO]{}: {} (certainty:{}) '.format(fname, y_hat_label, certainty))
            y_hat_sum

        # Export results
        self.export_predict_results(data, config['results_path'])


    def load_dataset_test(self, config):
        dataset = sulciDataset2.sulciDataset(csv_file = config['dataset_path'],
                                             root_dir = config['patches_root_dir'],
                                             dim = [32,64,64],
                                             crop_values = None)

        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        return dataset, dataLoader

    def load_model(self, config):
        
        ae = autoencoderResnet.AutoencoderRN()
        model = myNet.MyNet(
            pretrained = ae.encoder,
            h_size = 128,
            criterion = nn.CrossEntropyLoss(),
            dropout = 0,
            n_classes = 2,
            freeze = False)
        model = model.to(self.device)
        
        for param_name, param in model.named_parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=0, 
                               weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                         mode='min', 
                                                         verbose=True, 
                                                         patience=0, 
                                                         factor=0)
        
        model.set_optimizer(optimizer)
        model.set_scheduler(scheduler)

        model.load_from_best_model(os.path.normpath(
            os.path.join(config['models_dir'],
                         config['model_fname'])),
                         verbose=False)
        
        for g in optimizer.param_groups:
            g['lr'] = 0

        return model

    
    def export_predict_results(self, data, results_path):
        df_results= DataFrame(data)
        df_results.to_csv(os.path.normpath(results_path), 
                          index=False, 
                          header = ['Subject','DS','Certainty'])