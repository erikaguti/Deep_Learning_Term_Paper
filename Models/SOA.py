# Get Model
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class MNIST_SOA_Model():

    def __init__(self):
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.extractor = AutoFeatureExtractor.from_pretrained("farleyknight-org-username/vit-base-mnist")
        self.soa_model = AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")


    def run_inference(self, images):
        test_loss, correct = 0, 0
        incorrect_images = []
        
        with torch.no_grad():
            
            incorrect_images = []
            correct_images = []
            
            for image in range(len(images)):
        
                pred = self.soa_model(images[image]['image'])
        
                correct += (pred.logits.argmax(1) == images[image]['label']).type(torch.float).sum().item()

                if pred.logits.argmax(1) != images[image]['label']:
                    
                    incorrect_images.append(image)
                
                else:
                    correct_images.append(image)

        
        return correct/len(images), incorrect_images, correct_images
    

