from Data_Prep import inference_datasets, breakouts
from Models.SOA import MNIST_SOA_Model
from Data.Utils import transform_MNIST_tensor
from Models.denoiser_svhn_model import ConvDenoiser

# Check SOA Performance on clean MNIST

# transform pixels to SOA specifications

transformed_inference_datasets = [] 

for dataset in inference_datasets:

    dataset_copy = dataset.copy()
    
    transformed_images = []
    
    for image in dataset_copy:
        transformed_images.append(transform_MNIST_tensor(image))
    
    transformed_inference_datasets.append(transformed_images)


#for dataset in range(len(transformed_inference_datasets)):
correct_count, incorrect_images, correct_images = MNIST_SOA_Model().run_inference(transformed_inference_datasets[0])
print(correct_count * 100, '% Accuracy with breakout:', str(breakouts[dataset]))

# run incorrect_images through denoiser

## get back non-warped tensors:
















