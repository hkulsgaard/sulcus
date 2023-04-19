import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import autoencoder
import myNet
import glob
import myProject
from captum.attr import IntegratedGradients
import sulciDataset2
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import os.path
import copy

class projectSulcus(myProject.myProject):

    def __init__(self):        
        super(projectSulcus,self).__init__()

    def init_classifier(self, lr = 0.01, batch_size = 4, dim = [32,64,64], h_size = None,
                        patience = 5,  factor = 0.2, freeze = False, dropout = None,
                        train_path = None, validation_path = None, config_encoder = None,
                        config_decoder = None, ae_path = None, parches_dir = None, 
                        results_dir = None, crop=None, n_classes=None, verbose = True):

    # Loads data and the model for classification

        # Create results directory
        self.make_dir(results_dir)

        # Data loading
        self.load_dataset(train_path = train_path ,
                            validation_path = validation_path,
                            parches_dir = parches_dir,
                            batch_size = batch_size,
                            dim = dim,
                            transform = self.get_crop_transformation(dim,crop))

        self.load_autoencoder(config_encoder = config_encoder,
                                config_decoder = config_decoder,
                                dropout = dropout,
                                criterion = nn.BCELoss(), #FIX logits or not?
                                load_path = ae_path)
        
        self.load_classifier(h_size = h_size,
                            lr = lr,
                            patience = patience,
                            factor = factor,
                            dropout = dropout,
                            freeze = freeze,
                            n_classes = n_classes,
                            results_dir = results_dir)
        
        if verbose:
            self.print_summary(self.model, batch_size, dim, crop)

    def run(self, n_epochs = 10, results_dir = None, verbose=False):
        
        self.model.train_model(
            data_loaders = self.data_loaders,
            data_lengths = self.data_lengths,
            model_dir = results_dir,
            n_epochs = n_epochs,
            verbose = verbose)

        # Metrics plots
        self.build_plots(self.model, results_dir)

        # Prints the weights of the optimizer to check if was saved correctly
        #self.print_model_info('fc1','Last Saved State (fc1)') #for debugging
        
        self.print_finish()

    def run_tsne(self, dataset_path, parches_dir, dim, crop, h_path, h_size):
        
        dataset = sulciDataset2.sulciDataset(
            csv_file=dataset_path,
            root_dir=parches_dir,
            dim=dim,
            transform=self.get_crop_transformation(dim,crop))
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        X, y, patients = self.get_embeddings(h_path=h_path, h_size=h_size, dataset=dataset, dataLoader=dataLoader)
        
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

        '''

        if True:
            for i in range(0,5):
                print('[INFO]Building TSNE...')
                # We want to get TSNE embedding with 2 dimensions
                n_components = 2
                tsne = TSNE(n_components=n_components, learning_rate='auto', init='pca', perplexity=30)
                n_features = 32
                tsne_result = tsne.fit_transform(X[:,(n_features*i):(n_features*(i+1))])
                tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
                fig, ax = plt.subplots(1)            
                sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, legend='full', ax=ax, s=30, palette='deep')
        '''

        if False:
            # export TSNE data to an excel file
            excel_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y, 'patient':patients})
            excel_df.to_excel('prueba.xlsx')

        if False:
            # export embeddings
            excel_df = pd.DataFrame(X)
            excel_df.to_excel('embeddings.xlsx')

        plt.show()

    def load_autoencoder(self, config_encoder = None, config_decoder = None, dropout = 0.5,
            criterion = None, load_path = None):
        
        # Create empty autoencoder
        #criterion = nn.BCELoss()

        self.ae = autoencoder.Autoencoder(
            conf_file_encoder = config_encoder, 
            conf_file_decoder = config_decoder,
            prob_dropout = dropout,
            criterion = criterion)
        self.ae = self.ae.cuda()

        # Take encoder layers
        #encoder = autoencoder.Encoder(config_encoder,2,0)
        #encoder = encoder.cuda()

        # Optimizer
        optimizer = optim.Adam(list(self.ae.parameters()), lr=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True, factor=0.1, patience=5)
        self.ae.set_optimizer(optimizer)
        self.ae.set_scheduler(scheduler)

        self.ae.load_from_best_model(load_path)
        #pretrained_dict = pretrained_model.state_dict()
        #encoder_dict = encoder.state_dict()

        # Overwrite Encoder
        #pretrained_dict2 = {k: v for k, v in pretrained_dict.items() if k in encoder_dict} #BAD: encoder_dict no tiene los mismos items 
        #pretrained_dict = {k: v for k, v in pretrained_model.encoder.state_dict().items() if k in encoder_dict} #fix de la linea de arriba
        #encoder_dict.update(pretrained_dict) 
        #encoder.load_state_dict(encoder_dict)
    
    def load_classifier(self, h_size = None, lr = 0.1, patience = 5, factor = 0.1, 
                        dropout = None, freeze = None, n_classes = None, results_dir = None):
        
        self.model = myNet.MyNet(
            pretrained = copy.deepcopy(self.ae.encoder),
            h_size = h_size,
            #criterion = nn.CrossEntropyLoss(),
            #criterion = nn.BCEWithLogitsLoss(),
            criterion = nn.BCELoss(), # loss function
            dropout = dropout,
            n_classes = n_classes,
            freeze = freeze)
        self.model = self.model.cuda()

        optimizer = optim.Adam(list(self.model.parameters()), lr=lr, weight_decay=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            verbose=True,
            patience=patience,
            factor=factor)
        
        self.model.set_optimizer(optimizer)
        self.model.set_scheduler(scheduler)

        self.model.load_from_checkpoint(glob.glob(results_dir + "/checkpoint_epoch*.pt"))
        
        #self.print_model_info('fc1','Loaded State (fc1)') #for debugging
        
    def get_embeddings(self, h_path=None, h_size=None, dataset=None, dataLoader=None):
        if not os.path.exists(h_path):
            print('[INFO]Generating embeddings...')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = np.empty((0,h_size))
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
        
        else:
            print('[INFO]Loading embeddings from file...')
            h_file = pd.read_csv(h_path, header=None)

            #for i,j in h_file.iterrows():
            patients = h_file.iloc[:,0]
            y = h_file.iloc[:,1]
            X = h_file.iloc[:,2:]            

        return X, y, patients            

    def build_plots(self, model, dir):
        plt.figure(1)
        model.build_plot(
            data=model.losses,data_name='Loss',ylim=3,verbose=True,model_dir=dir)
        plt.figure(2)

        model.build_plot(
            data=model.aucs,data_name='AUC',ylim=1.1,verbose=True,model_dir=dir)
        plt.figure(3)

        model.build_plot(
            data=model.accuracies,data_name='Accuracy',ylim=1.1,verbose=True,model_dir=dir)
        plt.show()

    def print_model_info(self, attr, msg):
        if hasattr(self.model, attr):
            print('\n[DEBUGGING]{}'.format(msg))
            print(self.model.fc1.state_dict()) #comprobar si se guarda bien el modelo