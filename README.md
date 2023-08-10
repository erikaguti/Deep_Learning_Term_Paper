# Project Assessing Data Centric Approaches for DNN Model Performance Enhancement

This project was done in collaboration with [Shaney Sze](https://www.linkedin.com/in/shaneysze/) at the Barcelona School of Economics. Inspired by Andrew Ngo’s Campaign for [Data Centric AI](https://https-deeplearning-ai.github.io/data-centric-comp/), we set out to create a computer vision data ingestion pipeline that incorporates AI-informed data quality enhancement. A diagram of this pipeline created is outlined below:


<img width="830" alt="Pipeline Diagram" src="https://github.com/erikaguti/Deep_Learning_Term_Paper/assets/57955273/77b1b90f-ee48-4809-86eb-3deaa6b80c5f">


We decided to limit our scope to a Optical Character Recognition task. The original idea was to train a DNN to classify if an image had a distortion and if it did, send it to a denoiser DNN then send those images and the ones classified to not have a distortion to a SOA image classification model. However due to time constraints, we only created the denoiser DNN and used [vit-base-mnist](https://huggingface.co/farleyknight-org-username/vit-base-mnist) as the SOA OCR image classification model. A distorted image can be one that is blurry, has fog, is too bright, etc.

To test whether adding a denoiser DNN would improve the performance of the SOA model, a proof of concept was tested with a mix of 6,000 images from the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) and [Corrupted MNIST](https://www.tensorflow.org/datasets/catalog/mnist) datasets. 

From this proof of concept we found that the denoising improved the SOA OCR model performance by on average .53%

Below is an example of a batch of images used in the proof of concept and their denoised counterparts.

![n2](https://github.com/erikaguti/Deep_Learning_Term_Paper/assets/57955273/52ca6fa1-e9f9-4f67-a0d8-a187d55e2612)

As one more test we applied a denoising to images from [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/) and and digit classification performance. Again denoising improved performance, this time by 2.5%

A full write up of this project can be found in this [report](https://github.com/erikaguti/Deep_Learning_Term_Paper/blob/main/DataCentricDL.pdf).




