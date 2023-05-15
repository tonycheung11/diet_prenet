'''
    using this code to generate the test data we need before 
    running 'python main_resnet50.py ...'
'''
import shutil
import os
from collections import defaultdict

txt_dir = './meta_data/Food2k_label/test_finetune.txt'
data_txt = open(txt_dir,  'r')
imgs = []
label_record = defaultdict(int)
saving_path = './images_test_20000/'
for line in data_txt:
	line = line.strip()
	words = line.split(' ')
	label_ = int(words[1].strip())
	img_name = words[0]
	
	# choose the first 10 pictures as the test data for each label
	if label_ not in label_record or label_record[label_]<10:
		imgs.append(line + '\n')
		label_record[label_] += 1
		
		dir_name = words[0].split('/')[0]
		# make sure there is the directory we need
		if not os.path.exists(saving_path + dir_name):
			os.makedirs(saving_path + dir_name)
		# copy the image to the diretory we specify
		shutil.copy('/tmp/diet_datasets/Food2k_complete/'+img_name, saving_path+img_name)

# record the meta data of test images
with open('./meta_data/test_full_20000.txt', 'w') as f:
	f.writelines(imgs)
