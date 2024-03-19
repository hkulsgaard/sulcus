import numpy as np
import matplotlib.pyplot as plt

from os import path, makedirs
from abc import ABC, abstractmethod

from visdom import Visdom


class VisdomPlotter(object):
    # Visdom based generic plotter implementation

    def __init__(self, config):
        
        #super(GenericVisdomPlotter, self).__init__()
        self.id = ''

        # Environment for the plots and figures
        self.env = config['name']

        if 'hostname' in config:
            hostname = config['hostname']
        else:
            hostname = 'http://localhost/'

        if 'port' in config:
            port = int(config['port'])
        else:
            port = 8097

        # Initialize the object for visualization
        self.viz = Visdom(server=hostname, port=port)
        
        # Dictionary of figures (images)
        self.figures = dict()

        # Dictionary of plots (metrics)
        self.plots = dict()

        # Dictionary of text boxes (general info)
        self.texts = dict()
        

    def plot_value(self, plot_name, split_name, x, y, x_label='Epochs'):
        # Plot a line plot

        # if the plot is not in the dictionary, initialize one
        if (plot_name not in self.plots):
            self.plots[plot_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, 
                                                  opts=dict(legend=[str(split_name)],
                                                            title=plot_name,
                                                            xlabel=x_label,
                                                            ylabel=plot_name))
        # if the plot is already there, update
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, 
                          update='append', win=self.plots[plot_name], name=str(split_name))


    def plot_multiple_value(self, plot_name, x, y_values):
        # Plot multiple values within the same plot

        # get the split names
        split_names = y_values.keys()
        # iterate for each of them
        for split in split_names:
            # plot the values
            self.plot(plot_name, split, x, y_values[split])

    def display_image(self, image_key, image, opts):
        # Display given image in the plot

        # if the image is already in the plot, remove it to replace it for the new one
        if image_key in self.figures:
            self.viz.close(win=self.figures[image_key], env=self.env)
            del self.figures[image_key]

        # plot the image
        self.figures[image_key] = self.viz.image(image, env=self.env, opts=opts)

    def display_reconstruction(self, image_keys, imgs, captions):

        for imgage_key, img, caption in zip(image_keys, imgs, captions):
            self.display_image(imgage_key, np.rot90(img), dict(title=caption, width=350, height=350))

    def display_text(self, text, opts):
        #if (text_box not in self.texts):
        #    self.texts[text_box] = self.viz.text(text, env=self.env, opts=dict(title=text_box))
        #else:
        #    self.viz.text(text=text, win=self.texts[text_box], env=self.env, update='append')
        self.viz.text(text, env=self.env, opts=opts)

    def display_config(self, text_box, config, verbose=False):
        config_text = ''
        for key,value in zip(config.keys(),config.values()):
            if verbose: print(key,value)
            config_text = config_text + '<b>' + key + ':</b> ' + str(value) + '<br>'

        self.display_text(text=config_text, opts=dict(title=text_box, height=342))

    def set_id(self, id):
        self.id = id

'''
class ImageSegmentation2dVisdomPlotter(GenericVisdomPlotter):
    #Image Segmentation2d plotter

    def __init__(self, config):
        #Initializer
        super(ImageSegmentation2dVisdomPlotter, self).__init__(config)

        # retrieve the classes
        self.classes = string_to_list(config['architecture']['num-classes'])
        self.class_names= string_to_list(config['data']['classes'])
        self.all_classes = dict(zip(string_to_list(config['data']['dic-keys']), np.fromstring( config['data']['dic-values'], dtype=int, sep=',' )))


    def display_results(self, images, predictions, probabilities, true_labels, epoch):
        #Plot segmentation results
        
        fig = plt.figure()
        for i in range(0, len(images)):
            #Siempre me manejo con pil image. por eso tengo que pasarlo a np y transponer.
            np_img = np.array(images[i])   
            
            #El if porque las imagenes de ct no no son RGB.(nose xq)
            if (np_img.ndim < 3):
                np_img = np.stack((np_img,np_img,np_img),axis=0)
            else:
                np_img = np.transpose(np_img,(2, 0, 1))
            
            ent = entropy(probabilities[i], base=probabilities[i].shape[0], axis=0)
            ent = 255 * (ent - np.min(ent.flatten()))/(np.max(ent.flatten())- np.min(ent.flatten()))          
            prediction_rgb = self.segmentation_to_colors(predictions[i])
            true_labels_rgb = np.transpose(true_labels[i], (2,0,1))

            to_plot = np.stack((np_img, prediction_rgb, true_labels_rgb), axis=0)
            #stackear las probabilidades
            for k in range(probabilities[i].shape[0]):
                these_probabilities = np.stack((probabilities[i][k,:,:], probabilities[i][k,:,:], probabilities[i][k,:,:]),axis=0) * 255
                to_plot = np.append(to_plot, np.expand_dims(these_probabilities,axis=0), axis=0)                
            to_plot = np.append(to_plot,np.expand_dims(np.stack((ent,ent,ent),axis=0), axis=0), axis=0)
            self.display_image(str(i), to_plot)
                      
            #ax = fig.add_subplot(1, len(images), i+1, xticks=[], yticks=[])
            # display the image
            #plt.imshow(to_plot)            
        return fig


    def segmentation_to_colors(self, predictions):
        
        real_predictions =  np.zeros((predictions.shape[0],predictions.shape[1]),dtype=np.dtype('i'))
        for i in range(1,len(self.class_names)):
            real_predictions[predictions == i] = self.all_classes.get(self.class_names[i])
        
        return segm_to_colors(real_predictions)
'''