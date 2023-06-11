import copy
from Data_Prep import inference_datasets, breakouts
from Models.SOA import MNIST_SOA_Model
from Data.Utils import transform_MNIST_tensor
from Models.denoiser_svhn_model import ConvDenoiser
import torch
import tensorflow as tf

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

    X_noisy = [item['image'] for item in images_to_denoise]
    y_noisy = [item['label'] for item in images_to_denoise]
    
    X_clean = []
    y_clean = []
    for label in y_noisy:
        for image in images_not_to_denoise:
            if label.numpy() == image['label'].numpy():
                X_clean.append(image['image'])
                y_clean.append(image['label'])
                break

    # run incorrect_images through denoiser

    denoiser = ConvDenoiser(X_noisy, X_clean)

    history, score, predictions = denoiser.model(X_noisy, X_clean)

    # prep final run images
    final_images = []
    for image in range(len(incorrect_images)):
        prep = transformed_inference_datasets[dataset][incorrect_images[image]]
        prep['image'] = tf.convert_to_tensor(predictions[image])
        final = transform_MNIST_tensor(prep)
        final_images.append(prep)

    final_images.extend([transformed_inference_datasets[dataset][i] for i in correct_images])

    # post denoiser inference with SOA
    correct_count, incorrect_images, correct_images = MNIST_SOA_Model().run_inference(final_images)
    print(correct_count * 100, '% Accuracy with breakout:', str(breakouts[dataset]))
    post_denoiser_results.append((breakouts[dataset],correct_count * 100))

print("Pre Denoiser Results")
print(pre_denoiser_results)
print("Post Denoiser Results")
print(post_denoiser_results)
















