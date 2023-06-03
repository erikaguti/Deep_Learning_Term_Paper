import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot
import random

class Data():
    
    def __init__(self, split, distortions):
        self.sample_type = split
        self.distortion_list = distortions
        self.data_dict = self.load()
    
    def load(self):
        
        # get data
        
        mnist_clean = tfds.load('mnist', split=self.sample_type, shuffle_files=False, batch_size = -1)
        
        distorted_mnists = []
        
        for distortion in self.distortion_list:
            
            distorted_mnists.append(tfds.load(f'mnist_corrupted/{distortion}', 
                                              split=self.sample_type, 
                                              shuffle_files=False, 
                                              batch_size = -1))
        # store data in dictionary object
        
        data_dict = {'clean': []}
    
        for image in range(len(mnist_clean['image'])):
    
            data_dict['clean'].append({'id': f'clean_{image}', 
                               'image':mnist_clean['image'][image], 
                               'label':mnist_clean['label'][image],
                               'distortion': tf.constant(0)})

        for distortion in range(len(self.distortion_list)):
            
            data_dict.update({f'{self.distortion_list[distortion]}': []})
    
            for image in range(len(distorted_mnists[distortion]['image'])):
        
                data_dict[f'{self.distortion_list[distortion]}'].append({
                                                  'id':f'{self.distortion_list[distortion]}_{image}',
                                                  'image':distorted_mnists[distortion]['image'][image],
                                                  'label':distorted_mnists[distortion]['image'][image],
                                                  'distortion': tf.constant(distortion + 1)})
        return data_dict
    
    def display(self, image:np.ndarray, title:str='', cmap:str='gray', figsize=(5, 5)) -> None:
        
        '''Displays an image'''
        
        fig, ax = pyplot.subplots(1, figsize=figsize)
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=20)
        ax.set_axis_off()
        pyplot.tight_layout()
        pyplot.show()
        
    def dataset(self, size, ratios=dict(clean = 1)):
        
        images = []

        for key in ratios.keys():
            images.extend(random.choices(self.data_dict[key], k=int(ratios[key] * size)))
            
        return images

