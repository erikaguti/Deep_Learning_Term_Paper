import torch
import torch.nn.functional as F


def transform_MNIST_tensor(image):


        numpy_array = image['image'].numpy() / 255 # turn tensorflow tensor to numpy array and normalize pixel values
        if numpy_array.size == 150528:
                return image
        # move channel dimension to the front and convert into Pytorch tensor
        tensor = torch.tensor(numpy_array.reshape((1, 28, 28)))
    
        tensor = tensor.unsqueeze(0) # add batch size dimesion at index 0
    
        # expand image from 28x28 to 224x244
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
        # make the tensor have three channels instead of 1
        final_tensor = torch.cat((tensor, tensor, tensor), dim=1)
    
        image['image'] = final_tensor
    
        # convert label tensorflow tensor to Pytorch tensor
        image['label'] = torch.tensor(image['label'].numpy())

        return image
