from mmseg.apis import init_model, inference_model
import argparse
from skimage import io
import os
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

# Semantic Ids:
# name	             orig. id   pred. id   new id
# unlabeled	            0         0          0
# window_building	    1         1          1 
# window_parked	            2         2          2 
# window_transport	    3         3          3 
# traffic_light	            4         4          4 
# street_light_HT	    5         5          5 
# street_light_LT	    6         5          5 
# parked_front	            7         6          9 
# parked_rear	            8         7         10 
# moving_front	            9         6          9 
# moving_rear	           10         7         10  
# advertisement	           11         8         11  
# clock	                   12         9         13  
# inferred	           13         9         13  
# windows_group	           14         0          0  


# orig. id: The initial id that was assigned during the manual annotations of the light sources.
# pred. id: Since the net. predicts only a subset of the available light cat. these ids group some of the existent cat. (e.g. moving_front (9) and parked_front (7) to front (6))
# new id: For compatibility reasons with the orig. ids the pred ids are converted to the new ids.

# All this process is needed since:
# - pred ids must be a set of concecutive integer numbers.
# - the networks cannot predict all the available categories denoted by the orig. ids.

# from pred id to new id.
pred2new = OrderedDict({ 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 9:13, 7:10, 8:11, 6:9 })


def map_id(pred):
	for k in pred2new.keys():
		pred[pred == k] = pred2new[k]	
	return pred

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="This is a script performs inference and saves results.")
	parser.add_argument('img_dir', type=str, help="the path to directory where the images to perform inference on, are stored.")
	parser.add_argument('out_dir', type=str, help="the path to directory where the inferred semantic are stored.")
	parser.add_argument('config_path', type=str, help="the path to the config file used to train the model.")
	parser.add_argument('checkpoint_path', type=str, help="the path to the model's checkpoint.")
	args = parser.parse_args()
	
	config_path = args.config_path
	checkpoint_path = args.checkpoint_path
	img_dir = args.img_dir
	
	out_dir = args.out_dir
	os.makedirs(out_dir, exist_ok=True)

	model = init_model(config_path, checkpoint_path, device='cuda:0')
	
	for root, _, files in os.walk(img_dir):
		for name in tqdm(files):
			if not name.endswith('.png'):
				continue

			img_path = os.path.join(root, name)
	
			# get model's predictions.
			results = inference_model(model, img_path)
			pred = results.pred_sem_seg.data.cpu().numpy()[0]

			# map from pred id to new id.
			pred = map_id(pred)
			fn = os.path.basename(img_path)

			# store the result.
			out_path = os.path.join(out_dir, fn)
			io.imsave(out_path, pred.astype('uint8'), check_contrast=False)

