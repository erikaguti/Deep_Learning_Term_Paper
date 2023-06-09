
# Get MNIST Images

from Data.Data import DataLoader

distortions = ['shot_noise', 'motion_blur']  # desired distortions

test_data_obj = DataLoader('test', distortions)

# duplicates encoutered in create_dataset function so just using full testing dataset for now

test_data_dict = test_data_obj.load() # 10,000 per type of MNIST


MNIST_test_set_1 = test_data_dict['clean'][0:5000]

MNIST_test_set_2 = test_data_dict['clean'][5000:]

for distortion in distortions:
    MNIST_test_set_1.extend(test_data_dict[distortion][0:5000])
    MNIST_test_set_2.extend(test_data_dict[distortion][5000:])



