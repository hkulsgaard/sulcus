import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import autoencoder
import autoencoderResnet
import myNet
import glob
import myProject
import sulciDataset2
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import visualization
import copy
from captum.attr import IntegratedGradients
from captum.attr import Occlusion
import nibabel as nib
from pandas import DataFrame
import os

class projectSulcus(myProject.myProject):

    def __init__(self):        
        super(projectSulcus,self).__init__()
        self.models = list()
        self.best_epochs = list()

    def init_classifier(self, config, verbose = True):
        # Loads data and the model for classification

        # Create results directory
        self.results_dir = self.make_dir(config['classifier']['results_dir'],
                                         mode=config['classifier']['dir_mode'],
                                         cv=config['database']['cross_validation'])

        # Data loading
        self.load_dataset(config['database'])

        self.load_autoencoder(config['autoencoder'], criterion = nn.BCEWithLogitsLoss())
        
        self.load_classifier(config['classifier'])
        
        if verbose:
            self.print_summary(self.model,
                               config['classifier']['batch_size'],
                               config['database']['dim'],
                               config['database']['crop'])

    def train_classifier(self, config, verbose=False):
        
        if config['database']['cross_validation']:
            self.build_CV_splits_test(config['database'], export_folds=True)
            self.cv_splits = config['database']['cv_splits']
        else:
            self.cv_splits = 1

        for self.current_fold in range(0,self.cv_splits):
            self.init_classifier(config, verbose = False)

            self.plotter = visualization.VisdomPlotter(config['experiment'])
            self.plotter.display_config('Configuration', config['classifier'])

            self.model.train_model(
                data_loaders = self.data_loaders,
                data_lengths = self.data_lengths,
                model_dir = self.results_dir,
                n_epochs = config['classifier']['n_epochs'],
                plotter = self.plotter,
                verbose = verbose)

            # Metrics plots
            self.build_plots(self.model, self.results_dir, verbose=False)
        
        self.models.append(copy.deepcopy(self.model))
        self.best_epochs.append(self.model.last_saved_epoch)
        self.calculate_final_metrics()
        self.export_config(config)
        self.print_finish()

    def run_tsne(self, config):
        
        self.load_autoencoder(config['autoencoder'], criterion = nn.BCEWithLogitsLoss())

        dataset = sulciDataset2.sulciDataset(csv_file = config['database']['dataset_path'],
                                             root_dir= config['database']['parches_dir'],
                                             dim = config['database']['dim'],
                                             crop_values=config['database']['crop'])
        
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        X, y, patients = self.get_embeddings(config['autoencoder'],
                                             dataset=dataset,
                                             dataLoader=dataLoader)
        
        
        # debugging
        #print('Element: ', X.shape), 
        #print('Element: ', X[0].shape)
        #print('Label: ', y)
        #print('Element: ', X[0,0:100])

        print('[INFO]X shape:', X.shape)
        print('[INFO]y shape:', y.shape)

        print('[INFO]Building TSNE...')
        n_components = 2
        tsne = TSNE(n_components=n_components, learning_rate='auto', init='pca', perplexity=25)
        tsne_result = tsne.fit_transform(X)
        # (1000, 2) Two dimensions for each of our images
        
        # Plot the result of our TSNE with the label color coded
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
        fig, ax = plt.subplots(1)            
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=30, legend='full', palette='deep')

        # A lot of the stuff here is about making the plot look pretty and not TSNE
        #lim = (tsne_result.min()-5, tsne_result.max()+5)
        #ax.set_xlim(lim)
        #ax.set_ylim(lim)
        #ax.set_aspect('equal')
        #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        if False:
            # export TSNE data to an excel file
            excel_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y, 'patient':patients})
            excel_df.to_excel('prueba.xlsx')

        if False:
            # export embeddings
            excel_df = pd.DataFrame(X)
            excel_df.to_excel('embeddings.xlsx')

        plt.show()

    def load_autoencoder(self, config, criterion = None):
        
        # Create empty autoencoder
        #criterion = nn.BCELoss()
        ae_type = config.get('type')
        if (not ae_type==None) and ae_type=='resnet':
            self.ae = autoencoderResnet.AutoencoderRN(criterion=criterion)
        else:
            self.ae = autoencoder.Autoencoder(
                conf_file_encoder = config['config_encoder'], 
                conf_file_decoder = config['config_decoder'],
                criterion = criterion,
                h_size = config['h_size'],
                filters = config['filters'],
                pre_h_shape = config['pre_h_shape'])
            
        self.ae = self.ae.cuda()

        # Take encoder layers
        #encoder = autoencoder.Encoder(config_encoder,2,0)
        #encoder = encoder.cuda()

        # Optimizer
        optimizer = optim.Adam(list(self.ae.parameters()), lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True, factor=0.1, patience=5)
        
        self.ae.set_optimizer(optimizer)
        self.ae.set_scheduler(scheduler)

        self.ae.load_from_best_model(config['ae_path'])
        #pretrained_dict = pretrained_model.state_dict()
        #encoder_dict = encoder.state_dict()

        # Overwrite Encoder
        #pretrained_dict2 = {k: v for k, v in pretrained_dict.items() if k in encoder_dict} #BAD: encoder_dict no tiene los mismos items 
        #pretrained_dict = {k: v for k, v in pretrained_model.encoder.state_dict().items() if k in encoder_dict} #fix de la linea de arriba
        #encoder_dict.update(pretrained_dict) 
        #encoder.load_state_dict(encoder_dict)

        if config['export_encoder']:
            torch.save(self.ae.encoder, './resnet10_encoder_trained_ae64.pt')
    
    def load_classifier(self, config, best=False):
        
        self.model = myNet.MyNet(
            #pretrained = copy.deepcopy(self.ae.encoder),
            pretrained = self.ae.encoder,
            h_size = config['h_size'],
            criterion = nn.CrossEntropyLoss(),
            #criterion = nn.BCEWithLogitsLoss(),
            #criterion = nn.BCELoss(), # loss function
            dropout = config['dropout'],
            n_classes = config['n_classes'],
            freeze = config['freeze'])
        self.model = self.model.cuda()
        
        for param_name, param in self.model.named_parameters():
            param.requires_grad = True

        optimizer = optim.Adam(self.model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True, patience=config['patience'], factor=config['factor'])

        #optimizer = torch.optim.SGD(self.model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-3)
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        
        self.model.set_optimizer(optimizer)
        self.model.set_scheduler(scheduler)

        if best:
            self.model.load_from_best_model(config['model_path'], verbose=True)
        else:
            self.model.load_from_checkpoint(glob.glob(config['results_dir'] + "/checkpoint_epoch*.pt"))
            self.model.freeze_pretrained()
            #self.print_model_info('fc1','Loaded State (fc1)') #for debugging
        
        for g in optimizer.param_groups:
            g['lr'] = config['lr']
        
    def get_embeddings(self, config, dataset=None, dataLoader=None):
        
        print('[INFO]Generating embeddings...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = np.empty((0,config['h_size']))
        y = np.empty((0,1))
        patients = np.empty((0,1))

        for batch_id, (volume, label) in enumerate(dataLoader):
            # generates embeddings for the volumes
            volume = volume.to(device)
            h , _ = self.ae.predict(volume)
            
            # data format
            h = torch.flatten(h)
            h = h.cpu().detach().numpy()
            label = label.numpy()

            # stacks the embeddings and the labels
            X = np.vstack((X,h))
            y = np.vstack((y,label))
            patients = np.vstack((patients,dataset.get_fname(batch_id)))
        
        # save the generated embeddings
        embeddings = pd.DataFrame(np.hstack((patients,y,X)))
        #embeddings.to_csv(h_path, index=False, header=False)
        y = y[:,0]
        patients = patients[:,0]
        
        '''
        if not os.path.exists(h_path):
        else:
            print('[INFO]Loading embeddings from file...')
            h_file = pd.read_csv(h_path, header=None)

            #for i,j in h_file.iterrows():
            patients = h_file.iloc[:,0]
            y = h_file.iloc[:,1]
            X = h_file.iloc[:,2:]
        '''

        return X, y, patients

    def run_test(self, config):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Data loading
        test_dataset, dataLoader = self.load_dataset_test(config['database'])
        
        # Loading the best trained classifier
        if config['classifier']['model_type']=='delfi':
            self.load_delfi_model(config['classifier'])
        else:
            self.ae = autoencoderResnet.AutoencoderRN()
            self.load_classifier(config['classifier'], best=True) 
        
        #self.print_parameters(self.model, 'BEFORE_TEST', export=True)
        # Loop through the test images
        y_hats, labels, fnames, probs0, probs1 = list(), list(), list(), list(), list()
        
        self.model.eval()
        for batch_id, (volume, label) in enumerate(dataLoader):
            volume = volume.to(device)
            
            with torch.no_grad():
                probs = self.model(volume).cpu().detach()
            
            probs_loss, y_hat = torch.max(probs,1)
            fname = test_dataset.get_fname(batch_id)

            fnames.append(fname)
            labels.append(label.item())
            y_hats.append(y_hat.numpy().item())
            probs0.append(probs.numpy()[0][0])
            probs1.append(probs.numpy()[0][1])

            print('[INFO]{}: {}/{} (y/y_hat) ->probs: {:.3f} <> {:.3f}'.format(fname, label.item(), y_hat.item(), probs0[-1], probs1[-1]))  

        self.calculate_test_metrics(labels, y_hats)
        self.export_classification_results(fnames, labels, y_hats, probs0, probs1,
                                           os.path.split(config['classifier']['model_path'])[0],
                                           os.path.split(config['database']['dataset_path'])[1])
        
        '''
        #calibration curve for uncalibrated model
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(labels, np.array(all_probs)[:,1], n_bins=10)
        
        plt.plot(prob_pred, prob_true, color='#0b97e3', marker='o')
        plt.plot([0,0.5,1],[0,0.5,1], color='#aaaaaa', linestyle='dashed')
        plt.grid()
        plt.legend(['calibration curve','reference'])
        plt.xlabel('Prediction')
        plt.ylabel('Positive rate')
        plt.show()

        #Post-hoc Calibration for Classification
        from netcal.scaling import TemperatureScaling

        ground_truth = np.array(labels)
        confidences = np.array(all_probs)
        print(confidences)

        temperature = TemperatureScaling()
        temperature.fit(confidences, ground_truth)
        calibrated = temperature.transform(confidences)

        #Measuring Miscalibration for Classification
        from netcal.metrics import ECE

        n_bins = 10

        ece = ECE(n_bins)
        uncalibrated_score = ece.measure(confidences, ground_truth)
        calibrated_score = ece.measure(calibrated, ground_truth)

        print('[INFO] ECE unc:{} | ECE cal:{}'.format(uncalibrated_score,calibrated_score))
        
        #Visualizing Miscalibration for Classification
        from netcal.presentation import ReliabilityDiagram

        n_bins = 10

        diagram = ReliabilityDiagram(n_bins)
        a=diagram.plot(confidences, ground_truth)  # visualize miscalibration of uncalibrated
        b=diagram.plot(calibrated, ground_truth)   # visualize miscalibration of calibrated

        plt.show()
        '''
    def export_classification_results(self, fnames, labels, y_hats, probs0, probs1, results_dir, dataset_name):
        df_results= DataFrame(data={'fnames':fnames,
                                    'labels':labels,
                                    'y_hats':y_hats,
                                    'probs_0':np.round(probs0,3),
                                    'probs_1':np.round(probs1,3)})
        df_results.to_csv(results_dir + '/results_' + dataset_name, index=False)

    def calculate_test_metrics(self, labels, y_hats):
        from sklearn.metrics import f1_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import balanced_accuracy_score

        # Compute confusion matrix
        cm = confusion_matrix(labels, y_hats)
        (tn, fp), (fn, tp) = cm
        print(cm)      
        
        # Compute metrics
        accuracy = accuracy_score(y_true = labels, y_pred = y_hats)
        bacc = balanced_accuracy_score(y_true = labels, y_pred = y_hats) 
        f1 = f1_score(y_true = labels, y_pred = y_hats) 
        tn, fp, fn, tp = confusion_matrix(labels, y_hats).ravel()
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        precision = precision_score(y_true = labels, y_pred = y_hats) 
        
        # Print metrics
        #print('[INFO]Accuracy: {:.3f}'.format(accuracy))
        print('[INFO]Balanced Accuracy: {:.3f}'.format(bacc))
        print('[INFO]F1 Score: {:.3f}'.format(f1))
        print('[INFO]Sensitivity: {:.3f}'.format(sensitivity))
        print('[INFO]Specificity: {:.3f}'.format(specificity))
        print('[INFO]Precision: {:.3f}'.format(precision))

    def build_plots(self, model, dir, verbose=False):
        plt.figure(1)
        model.build_plot(
            data=model.losses,data_name='Loss',ylim=1,model_dir=dir,verbose=verbose)
        plt.figure(1).clear()
        
        plt.figure(2)
        model.build_plot(
            data=model.accuracies,data_name='Accuracy',ylim=1.1,model_dir=dir,verbose=verbose)
        plt.figure(2).clear()

        plt.figure(3)
        model.build_plot(
            data=model.baccs,data_name='Balanced Accuracy',ylim=1.1,model_dir=dir,verbose=verbose)
        plt.figure(3).clear()

        plt.figure(4)
        model.build_plot(
            data=model.f1s,data_name='F1', ylim=1.1, model_dir=dir,verbose=verbose)
        plt.figure(4).clear()
            

    def print_model_info(self, attr, msg):
        if hasattr(self.model, attr):
            print('\n[DEBUGGING]{}'.format(msg))
            print(self.model.fc1.state_dict()) #comprobar si se guarda bien el modelo

    def show_weight_test(self, msg='resnet encoder weight sample'):
        print('[INFO]{}'.format(msg))
        print(self.ae.encoder.module.layer1[0].conv1.weight[0,0,0,:,:])

    def calculate_final_metrics(self):
        f1_models = np.array([])
        acc_models = np.array([])
        bacc_models = np.array([])
        for m, best_epoch in zip(self.models,self.best_epochs):
            best_epoch = best_epoch-1
            f1_models = np.append(f1_models, m.f1s['val'][best_epoch])
            acc_models = np.append(acc_models, m.accuracies['val'][best_epoch])
            bacc_models = np.append(bacc_models, m.baccs['val'][best_epoch])
        
        print('\n[INFO]Final average F1 scores: ', np.mean(f1_models))
        print('[INFO]Final average accuracies: ', np.mean(acc_models))
        print('[INFO]Final average balanced accuracies: ', np.mean(bacc_models))


    def run_captum(self, config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        results_dir = self.make_dir(config['captum']['export_path'], overwrite=True)

        # Data loading
        dataset, dataLoader = self.load_dataset_test(config['database'])
        
        # Loading the best trained classifier
        if config['classifier']['model_type']=='delfi':
            self.load_delfi_model(config['classifier'])
        else:
            self.ae = autoencoderResnet.AutoencoderRN()
            self.load_classifier(config['classifier'], best=True)
        
        # Set the seed for aleatorization for captum
        torch.manual_seed(config['captum']['seed'])
        np.random.seed(config['captum']['seed'])

        # number of the slice to display
        slice_num = config['captum']['slice_num']
        img_range = config['captum']['img_range']
        captum_class = config['captum']['captum_class']
        pdim = self.get_patch_dim(config['database']['dim'],
                                  config['database']['crop'])

        print('[INFO]Captuming...')
        self.model.eval()
        for batch_id, (volume, label) in enumerate(dataLoader):
            volume = volume.to(device)
            probs = self.model(volume).cpu().detach()
            probs_loss, y_hat = torch.max(probs,1)
            
            if (batch_id>=img_range[0]) and (batch_id<=img_range[1]):
                #Abro una imagen, la paso por autoencoder y calculo la diferencia entre ambas
                fname = dataset.get_fname(batch_id)

                baseline = torch.zeros(volume.shape).to(device)
                if (config['captum']['mode']=='occlusion'):
                    ablator = Occlusion(self.model)
                elif (config['captum']['mode']=='ig'):
                    ig = IntegratedGradients(self.model)

                for c in range(captum_class[0],captum_class[1]+1):
                    if (config['captum']['mode']=='occlusion'):
                        attributions = ablator.attribute(volume, target=c, sliding_window_shapes=(1,3,3,3))
                    elif (config['captum']['mode']=='ig'):
                        attributions, delta = ig.attribute(volume, baseline, target=c, return_convergence_delta=True)                   

                    attri = attributions.view(-1, pdim[0], pdim[1], pdim[2]).cpu()

                    #save the attributions map
                    if slice_num=='all':
                        new_fname = results_dir + '/captum_{}_[class{}]'.format(fname, c)
                        nii_img = nib.Nifti1Image(attri.view(pdim[0], pdim[1], pdim[2]).numpy(), affine=np.eye(4))
                        nib.save(nii_img, new_fname)
                        
                        print("[INFO] Image: {}".format(fname))

                    else:
                        slice_0 = attri[0, slice_num, :, :]
                        img = volume.view(-1,pdim[0],pdim[1],pdim[2]).cpu()
                        img_orig = img[0, slice_num, :, :]

                        print("\n[INFO] Image: {}".format(fname))
                        print("[INFO] Captum image value range from {:.3f} to {:.3f}".format(torch.min(slice_0).item(),torch.max(slice_0).item()))
                        print("[INFO] Original image value range from {:.3f} to {:.3f}".format(torch.min(img_orig).item(),torch.max(img_orig).item()))
                        print("[INFO] Probabilities: {}".format(probs))

                        self.show_slices(
                            [slice_0, img_orig], ["bwr", "gray"], 'Image:{}/ Class:{}/ y:{}/ y_hat:{}'.format(batch_id, c, label.item(), y_hat.item()))
                            #[slice_0, slice_0*img_orig, img_orig], ["bwr", "bwr", "gray"], 'Patient {}/ h {}'.format(batch_id, h_i))

                        if config['captum']['export_images']: 
                            plt.savefig(results_dir + '/captum_{}_[class{}_slice{}]'.format(fname, c, slice_num))

                        if config['captum']['verbose']:
                            plt.show()
                
                #save the original input image (for comparison purpouse)
                if slice_num=='all':
                    orig_fname = results_dir + '/{}'.format(fname)
                    nii_orig = nib.Nifti1Image(volume.view(pdim[0], pdim[1], pdim[2]).cpu().numpy(), affine=np.eye(4))
                    nib.save(nii_orig, orig_fname)
                    print(fname)

        print('\n[INFO]Job done!\n')


    def load_dataset_test(self, config):
        dataset = sulciDataset2.sulciDataset(csv_file = config['dataset_path'],
                                             root_dir = config['parches_dir'],
                                             dim = config['dim'],
                                             crop_values = config['crop'])

        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        return dataset, dataLoader
    
    def load_delfi_model(self, config):
        import argparse
        from delfi import model as model_delfi

        sets = argparse.ArgumentParser()
        sets.model = 'resnet'
        sets.model_depth = 10
        sets.input_W = 448
        sets.input_H = 448
        sets.input_D = 56
        sets.resnet_shortcut = 'B'
        sets.no_cuda = False
        sets.n_seg_classes = 2
        sets.dropout = 0.1
        sets.gpu_id = [0]
        sets.phase = 'test'
        sets.target_type = "normal"
        
        self.model, params = model_delfi.generate_model(sets)
        self.model.cuda()
        checkpoint = torch.load(config['model_path'])
        self.model.load_state_dict(checkpoint['state_dict'])

    def print_parameters(self, model, model_name="Unknown", export=False):
        print('\n[INFO]Printing parameters from model {}'.format(model_name))
        lines, p_names = list(), list()

        print('[INFO]Names:')
        for name, param in model.named_parameters():
            print(name)
            p_names.append(name)

        print('[INFO]Values:')
        for name, param in model.named_parameters():
            line = '{} >>> {}'.format(name, param)
            print(line)
            lines.append(line)
        
        if export:
            fname = './parameters/' + model_name + '_params.txt'
            with open(fname, 'w') as f:
                f.write(f"[INFO]Names:\n")
                for name in p_names:
                    f.write(f"{name}\n")

                f.write(f"\n[INFO]Values:\n")
                for line in lines:
                    f.write(f"\n{line}\n")
