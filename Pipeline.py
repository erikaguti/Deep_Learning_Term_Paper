import copy
from Data_Prep import inference_datasets, breakouts
from Models.SOA import MNIST_SOA_Model
from Data.Utils import transform_MNIST_tensor
from Models.denoiser_svhn_model import ConvDenoiser
import torch

pre_denoiser_results = []
post_denoiser_results = []

# Check SOA Performance on clean MNIST

# transform pixels to SOA specifications

transformed_inference_datasets = [] 

for dataset in inference_datasets:

    dataset_copy = copy.deepcopy(dataset)
    
    transformed_images = []
    
    for image in dataset_copy:
        transformed_images.append(transform_MNIST_tensor(image))
    
    transformed_inference_datasets.append(transformed_images)



for dataset in range(len(transformed_inference_datasets)):
    
    # inital inference with SOA
    correct_count, incorrect_images, correct_images = MNIST_SOA_Model().run_inference(transformed_inference_datasets[dataset])
    print(correct_count * 100, '% Accuracy with breakout:', str(breakouts[dataset]))
    pre_denoiser_results.append((breakouts[dataset],correct_count * 100))


    # get back non-warped tensors:

    images_to_denoise = [inference_datasets[dataset][i] for i in incorrect_images]
    images_not_to_denoise = [inference_datasets[dataset][i] for i in correct_images]


    # run incorrect_images through denoiser

    X_noisy = [item['image'] for item in images_to_denoise]
    y_noisy = [item['label'] for item in images_to_denoise]

    X_clean = [item['image'] for item in images_not_to_denoise]
    y_clean = [item['label'] for item in images_not_to_denoise]

    denoiser = ConvDenoiser(X_noisy, X_clean, y_noisy, y_clean)

    history, loss, acc, predictions = denoiser.model(X_noisy, X_clean, y_noisy, y_clean)


    # prep final run images
    final_images = []
    for image in range(len(incorrect_images)):
        prep = transformed_inference_datasets[0][image]
        prep['label'] = torch.tensor(predictions.argmax(1)[image])
        final_images.append(prep)

    final_images.extend([transformed_inference_datasets[0][i] for i in correct_images])

    # post denoiser inference with SOA
    correct_count, incorrect_images, correct_images = MNIST_SOA_Model().run_inference(final_images)
    print(correct_count * 100, '% Accuracy with breakout:')
    post_denoiser_results.append((breakouts[dataset],correct_count * 100))
















