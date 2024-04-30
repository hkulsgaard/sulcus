import torch
from torch import nn as nn
from torch import optim as optim
import autoencoderResnet
import myNet
import myProject
import sulciDataset2
from torch.utils.data import DataLoader
from pandas import DataFrame
from captum.attr import Occlusion
import nibabel as nib
import utils
import numpy as np
import os

class projectFtencoder(myProject.myProject):

    def __init__(self):        
        super(projectFtencoder,self).__init__()
        self.models = list()

    def run_predict(self, config):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ablators = list()
        try: 
            config['occlusion_maps']
        except:
            config['occlusion_maps'] = False

        # Data loading
        dataset, dataLoader = self.load_dataset_test(config)

        # Loading the Ft-encoders
        for model_fname in config['models_fnames']:
            config['model_fname'] = model_fname
            model = self.load_model(config)
            model.eval()
            self.models.append(model)
            if config['occlusion_maps']: ablators.append(Occlusion(model))
            print('[INFO]Loaded trained model: "{}"'.format(model_fname))
        
        print()
        n_models = (len(self.models))

        # Initialize data dict
        data={'fnames':list(),
              'y_hats':list(),
              'certainty':list()}
        
        # Loop through the images
        for batch_id, (volume, label) in enumerate(dataLoader):
            y_hats=list()
            fname = dataset.get_fname(batch_id)
            volume = volume.to(device)
                        
            for model in self.models:
                with torch.no_grad():
                    probs = model(volume).cpu().detach()
                probs_loss, y_hat = torch.max(probs,1)
                y_hats.append(y_hat.numpy().item())

            if config['occlusion_maps']: 
                self.generate_occlusion_maps(volume= volume, 
                                            metadata= dataset.get_metadata(batch_id),
                                            y_hats =y_hats, 
                                            ablators= ablators)
            
            # Majority voting
            y_hat_sum = sum(y_hats)
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
        
    def generate_occlusion_maps(self, volume, metadata, y_hats, ablators):
        print('[WARNING]Occlusion map generation could take at least 1 hour per patch')
        for (y_hat, ablator, k) in zip(y_hats, ablators, range(1,len(y_hats)+1)):
            print('[INFO]Generating occlusion map k{}...'.format(k))
            # Calculate the occlusion map
            attributions = ablator.attribute(volume, target=y_hat, sliding_window_shapes=(1,3,3,3))
            omap = attributions.view(-1, 32, 64, 64).cpu()

            # Save the occlusion map
            omap_img = nib.Nifti1Image(omap.view(32, 64, 64).numpy(), affine=metadata['affine'], header=metadata['header'])
            #omap_img = nib.Nifti1Image(omap.view(32, 64, 64).numpy(), affine=np.eye(4))
            #nib.save(omap_img, utils.addSufix(utils.replaceDir(metadata['path'], results_dir),'_[omap_ftek{}_{}]'.format(k,y_hat)))
            nib.save(omap_img, utils.addSufix(metadata['path'],'_[omap_ftek{}_DS{}]'.format(k,y_hat)))

            # Save the same occlusion map but splited into positive and negative attribution NIFITS
            omap_pos = omap.view(32, 64, 64).numpy()
            omap_pos[omap_pos<0] = 0
            omap_pos_nii = nib.Nifti1Image(omap_pos, affine=metadata['affine'], header=metadata['header'])
            nib.save(omap_pos_nii, utils.addSufix(metadata['path'],'_[omap_positive_ftek{}_DS{}]'.format(k,y_hat)))

            #omap_neg = (omap.view(32, 64, 64).numpy())*-1
            #omap_neg[omap_neg<0] = 0
            #omap_neg_nii = nib.Nifti1Image(omap_neg, affine=metadata['affine'], header=metadata['header'])
            #nib.save(omap_neg_nii, utils.addSufix(metadata['path'],'_[omap_negative_ftek{}_DS{}]'.format(k,y_hat)))
