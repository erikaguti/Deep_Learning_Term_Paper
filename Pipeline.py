from Data_Prep import MNIST_test_set_1, MNIST_test_set_2
from Models.SOA import MNIST_SOA_Model
from Data.Utils import transform_MNIST_tensor

# Check SOA Performance on clean MNIST

transformed_MNIST_test_set_1 = []
for image in MNIST_test_set_1:
    transformed_MNIST_test_set_1.append(transform_MNIST_tensor(image))

test_run_1_results, incorrect_images = MNIST_SOA_Model().run_inference(transformed_MNIST_test_set_1)

print(test_run_1_results * 100, '% Accuracy')

# run incorrect_images through denoiser

## get back non-warped tensors:













