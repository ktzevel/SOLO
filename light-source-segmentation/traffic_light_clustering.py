"""
	The purpose of this code is to process the light annotations and replace the traffic_light class
	with one of the following classes:

		- traffic_light_red
		- traffic_light_green
		- traffic_light_orange
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb, lab2xyz, xyz2rgb, xyz2lab
from skimage.measure import label
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import os
import argparse

import labels


"""
	ann_dir should have the following structure:
		ann_dir/
			gt/{fid}.png
			rgb/{fid}.png

	All images/masks should be contained in this directory,
	both the annotated ones and the predicted ones.
"""
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This script classifies further the traffic lights to red, green and orange.")
	parser.add_argument('ann_dir', type=str, help="the path to the light annotations directory.")
	parser.add_argument('out_dir', type=str, help="the path to the output directory.")
	args = parser.parse_args()

	out_dir = args.out_dir
	os.makedirs(out_dir, exist_ok=True)

	ann_dir = args.ann_dir

	trl_sem_ids = [
		  labels.ls.name2id['traffic_light_green']
		, labels.ls.name2id['traffic_light_orange']
		, labels.ls.name2id['traffic_light_red']
	]

	# those values are from the night time illuminants dataset.
	ref_xyz = [
		  [0.05385712952877754, 0.07979921473723006, 0.05258982041222988 ] # Green
		, [0.10410573480865178, 0.08018557595106497, 0.009918626925243831] # Orange
		, [0.12193108095233852, 0.06189343924936608, 0.008806302441526996] # Red
	]

	trl_id = labels.ls.name2id['traffic_light']

	gt_l = []
	rgb_l = []
	for root, _, files in os.walk(ann_dir, topdown=False):
		for name in files:

			if not name.endswith('.png'):
				continue

			if 'rgb' in root:
				rgb_l.append(os.path.join(root, name))

			elif 'gt' in root:
				gt_l.append(os.path.join(root, name))

	# make sure they are sorted the same way.
	rgb_l = sorted(rgb_l)
	gt_l = sorted(gt_l)

	lab_color_l = []
	pairs = list(zip(rgb_l, gt_l))

	# Kmeans
	# find centroids.
	for rgb_path, gt_path in tqdm(pairs, dynamic_ncols=True, desc='Calculating centroids...'):

		# To find the centroids use only the manually created annotations.
		name = os.path.basename(rgb_path)
		fid = name.split('.')[0] # drop extension.
		fid = '_'.join(fid.split('_')[:3]) # drop anything after fid.
		if fid not in labels.manual_annotations:
			continue

		rgb = np.array(Image.open(rgb_path))
		gt = np.array(Image.open(gt_path))

		# get pixels belonging to the traffic light class.
		m = (gt == trl_id)

		# handle the case where more than one traffic lights appear.
		regs, num = label(m, return_num=True)
		for i in range(num):

			m = (regs == i+1)
			bs = find_boundaries(m, mode='inner')
			
			# drops the boundaries of the mask to account for non-precise annotations.
			m = (m & ~bs)
			pxs = rgb[m]

			# pixels lower bound to sample the color from.
			if pxs.shape[0] < 4:
				continue

			# convert to lab space (euclidean space).
			# assuming sRGB with D65.
			pxs = rgb2lab(pxs, illuminant='D65')
			
			avg_color = np.mean(pxs, axis=0)

			lab_color_l.append(avg_color)

	lab = np.array(lab_color_l)

	# drop the intensity component L.
	ab = lab[:, 1:]

	# pca = PCA(n_components=2)
	# pca.fit(ab)
	# ab = pca.transform(ab)
	kmeans = KMeans(n_clusters=len(trl_sem_ids), random_state=0, n_init="auto").fit(ab)

	centers = kmeans.cluster_centers_
	labels = kmeans.labels_

	# disp = DecisionBoundaryDisplay.from_estimator(kmeans, ab, response_method="predict", xlabel='a', ylabel='b',alpha=0.3, cmap="Greys")
	# a, b = ab.T
	# disp.ax_.scatter(a, b, c=lab2rgb(lab), edgecolor='black')
	# disp.ax_.scatter(centers[:,0], centers[:,1], c='gold', edgecolor='black', marker='*')
	# plt.show(block=False)

	# ref_preds: trl_id -> pred_id
	ref_preds = kmeans.predict(xyz2lab(ref_xyz)[:,1:])

	# pred2trl: pred_id -> trl_id
	pred2trl = np.argsort(ref_preds)
	
	# substitute traffic_light class with the predicted label.
	for rgb_path, gt_path in tqdm(pairs, dynamic_ncols=True, desc='Generating new annotations...'):

		rgb = np.array(Image.open(rgb_path))
		gt = np.array(Image.open(gt_path))

		# get pixels belonging to the traffic light class.
		m = (gt == trl_id)

		# handle the case where more than one traffic lights appear.
		regs, num = label(m, return_num=True)
		new_mask = gt.copy()
		for i in range(num):

			m = (regs == i+1)
			pxs = rgb[m]

			# convert to lab space (euclidean space).
			# assuming sRGB with D65.
			pxs = rgb2lab(pxs, illuminant='D65')
			
			lab = np.mean(pxs, axis=0)
			ab = lab[1:]
			ab = np.expand_dims(ab, axis=0)
			pred = kmeans.predict(ab)[0]
			l = trl_sem_ids[pred2trl[pred]]

			# change the label accordingly
			new_mask[m] = l

		# save new annotation
		fn = os.path.basename(rgb_path)
		out_path = os.path.join(out_dir, fn)
		new_gt = Image.fromarray(new_mask)
		new_gt.save(out_path)
		

	# visualize results:
	# gt_l = []
	# rgb_l = []
	# for root, _, files in os.walk(ann_dir, topdown=False):
	# 	for name in files:

	# 		if not name.endswith('.png'):
	# 			continue

	# 		if 'rgb' in root:
	# 			rgb_l.append(os.path.join(root, name))

	# 		elif 'gt' in root and 'new' in name:
	# 			gt_l.append(os.path.join(root, name))

	# # make sure they are sorted the same way.
	# rgb_l = sorted(rgb_l)
	# gt_l = sorted(gt_l)

	# lab_color_l = []
	# pairs = list(zip(rgb_l, gt_l))
	# fig, ax = plt.subplots()
	# for rgb_path, gt_path in tqdm(pairs, dynamic_ncols=True):
	# 	rgb = np.array(Image.open(rgb_path))
	# 	gt = np.array(Image.open(gt_path))

	# 	m = (gt == 21) | (gt == 22) | (gt == 23)
	# 	if not m.any():
	# 		continue

	# 	ax.clear()
	# 	ax.imshow(rgb)
	# 	plt.show(block=False)
	# 	ax.imshow(gt, alpha=0.5)
	# 	ax.set_title(f"Image:{os.path.basename(rgb_path)}")
	
	# 	plt.pause(10)