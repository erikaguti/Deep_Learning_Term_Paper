
# Get MNIST Images

from Data.Data import DataLoader

distortions = ['shot_noise', 'motion_blur', 'fog']  

test_data_obj = DataLoader('test', distortions)

# create datasets to run inference on various data quality breakouts

inference_datasets = []

size = 10000

breakouts = [{'clean': 1}] # only clean

for distortion in distortions:
    breakouts.append({'clean': .5, distortion : .5}) # to answer: which distortion decreases performance the most?

breakouts.append({'clean':.25, 'shot_noise': .25, 'motion_blur': .25, 'fog': .25}) # everything

for breakout in breakouts:
   inference_datasets.append(test_data_obj.create_dataset(size, breakout))
