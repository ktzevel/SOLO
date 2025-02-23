from cct_utils import xyz2cct
import selector
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import pickle
from tqdm import tqdm
import colour.plotting as cp
import argparse
import shutil

from typing import List, Any

from skimage.color import xyz2lab, lab2xyz

import camera_pipeline
from camera_pipeline import xyz_to_srgb


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()

		return json.JSONEncoder.default(self, obj)


def compute_mean_XYZ(img:np.ndarray, m:np.ndarray)->list:
	""" 
		Computes the mean XYZ coordinates.
		m (np.ndarray): the selected region.
	"""
	# convert to euclidean space first.
	img = xyz2lab(img, illuminant='E')

	# takes the mean XYZ coord in the selected region, indicated by the mask m.
	mean_xyz = np.mean(img, where=m[..., None], axis=(0,1))

	mean_xyz = lab2xyz(mean_xyz, illuminant='E')

	return mean_xyz.tolist()
	# return np.round(mean_xyz, decimals=3).tolist()

def xyY2xyz(xyY:np.ndarray)->list:
	""" 
		Computes XYZ from xyY.
		X = xY / y
		Y = Y
		X = (1 - x - y)Y / y
	"""
	def protect(val):
		if val >= 1.0:
			return 0.999999
		if val <= 0.0:
			return 0.000001

		return val

	x, y, Y = xyY
	x = protect(x)
	y = protect(y)
	Y = protect(Y)
	return x * Y / y, Y, (1 - x - y) * Y / y

def xyz2xy(xyz:np.ndarray)->list:
	""" 
		Computes the xy chromaticity coordinates.
		x = X / (X + Y + Z)
		y = Y / (X + Y + Z)
	"""
	return xyz[:2]/ np.sum(xyz)

def get_Y(img:np.ndarray):
	return img[..., 1]

def compute_maxY(img:np.ndarray, m:np.ndarray):
	""" Computes the luminocity. """
	Y = img[..., 1]
	maxY = np.max(Y, where=m, initial=-1)
	return maxY

# TODO: maxY should be calculated on a region of pixels, not on a single one.
def compute_argmaxY(img:np.ndarray, mask:np.ndarray, maxY):
	Y = get_Y(img)
	return np.unravel_index(np.argmax(mask & (Y == maxY)), Y.shape)


def process_images(paths:List[str], ckpt_dir:str):
	
	print('Running the image processing pipeline...')
	print('Stand by to select the regions of interest afterwards.')

	def get_category(path:str)->str:
		dir_path = os.path.dirname(p)
		dir_name = dir_path.split(os.path.sep)[-1]
		return dir_name

	proc_names = []
	for fn in os.listdir(ckpt_dir):
		if os.path.isfile(os.path.join(ckpt_dir, fn)):
			proc_names.append(fn.split('-')[1])

	for p in tqdm(paths):
		name = os.path.basename(p).split('.')[0]
		cat = get_category(p)

		# check if the image is already processed.
		if name in proc_names:
			fn = [fn for fn in os.listdir(ckpt_dir) if fn.endswith(name)][-1]
			new_fn = '-'.join([cat, name])
			if os.path.isfile(os.path.join(ckpt_dir, new_fn)):
				continue

			# opens the corresponding pickle, change the category and save under different name.
			new_path = os.path.join(ckpt_dir, new_fn)
			old_path = os.path.join(ckpt_dir, fn)
			d = None
			with open(old_path, 'rb') as fp:
				d = pickle.load(fp)
			
			d['cat'] = cat
			with open(new_path, 'wb') as fp:
				pickle.dump(d, fp)

			continue

		img, meta = camera_pipeline.single_image(p)
		d = {
				  'name': name
				, 'cat': cat
				, 'img': img
				, 'meta': meta
			}

		fn = '-'.join([d['cat'], d['name']])
		with open(os.path.join(ckpt_dir, fn), 'wb') as fp:
			pickle.dump(d, fp)
		
		proc_names.append(name)

def compute_props(ckpt_dir:str):

	print('Computing image properties...')
	ckpt_list = os.listdir(ckpt_dir)
	mask_dir = os.path.join(ckpt_dir, 'masks') 
	os.makedirs(mask_dir, exist_ok=True)

	props_l = []
	for ckpt_name in tqdm(ckpt_list):

		path = os.path.join(ckpt_dir, ckpt_name)
		if not os.path.isfile(path):
			continue

		d = None
		with open(path, 'rb') as fp:
			d = pickle.load(fp)

		img = d['img']
		name = d['name']
		cat = d['cat']
		meta = d['meta']

		# some categories are sharing dataset samples.
		# keeps the part of the name referring to the unique dataset sample.
		ckpt_name = ckpt_name.split('-')[1]

		mask_path = os.path.join(mask_dir, ckpt_name + '.npy')
		if os.path.exists(mask_path):
			mask = np.load(mask_path)
		else:
			mask = selector.select_region(xyz_to_srgb(img), *img.shape[:2])
			np.save(mask_path, mask)
		
		props = {}
		props['name'] = name
		props['cat'] = cat
		props['XYZ'] = compute_mean_XYZ(img, mask)
		props['xy'] = xyz2xy(props['XYZ'])
		Y = [0.01] # intensity value for xyY. Make all samples equal in terms of intensity.
		xyY = list(props['xy']) + Y
		props['XYZ_no_illum'] = xyY2xyz(xyY)
		props['maxY'] = compute_maxY(img, mask)
		props['argmaxY'] = [int(i) for i in compute_argmaxY(img, mask, props['maxY'])]


		props['cct'] = round(xyz2cct(props['XYZ']))
		props['iso'] = meta['iso']
		props['exposure_time'] = round(meta['exposure_time'], 3)
		props['aperture'] = round(meta['aperture'], 3)
		props['focal_length'] = round(meta['focal_length'], 3)
		props_l.append(props)

	return props_l

def plot_chromaticities(plops_l, ckpt_dir, save_dir, slots=11, dpi=250):

	save_dir_base = save_dir
	for i, prop in enumerate(props_l):

		d = None
		fn = '-'.join([prop['cat'], prop['name']])
		with open(os.path.join(ckpt_dir, fn), 'rb') as fp:
			d = pickle.load(fp)
		
		img = d['img']
		cat = prop['cat']

		save_dir = os.path.join(save_dir_base, cat)
		img_save_path = os.path.join(save_dir, fn + '_img.jpg')
		col_save_path = os.path.join(save_dir, fn + '_pallete.jpg')
		xy_save_path = os.path.join(save_dir, fn + '_xy.jpg')
		if os.path.exists(img_save_path) or os.path.exists(col_save_path) or os.path.exists(xy_save_path):
			continue


		os.makedirs(save_dir, exist_ok=True)
		f = plt.figure()
		plt.rcParams.update({'font.size': 18})

		# gs = matplotlib.gridspec.GridSpec(1,2)
		# ax00 = f.add_subplot(gs[0, 0])
		# ax01 = f.add_subplot(gs[0, 1])
		# ax10 = f.add_subplot(gs[1, 0])
		# ax11 = f.add_subplot(gs[1, 1])

		f, ax = cp.plot_chromaticity_diagram_CIE1931(show=False, tight_layout=True) # CIE 1931 2 Degree Standard Observer.
		vec = np.round(prop['xy'], 2).tolist()
		ax.plot(*prop['xy'], marker="*", markersize=10, alpha=1, markeredgecolor="black", markerfacecolor="yellow", label=f"({vec[0]},{vec[1]})")
		ax.legend()
		# ax.set_title('Chromaticity diagram CIE 1931', fontsize='x-large')
		ax.set_xlim(-.2, 1)
		ax.set_ylim(-.2, 1)
		ax.set_xlabel('')
		ax.set_ylabel('')
		plt.xticks([])
		plt.yticks([])
		plt.savefig(xy_save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
		plt.close()

		plt.figure()
		plt.xticks([])
		plt.yticks([])
		plt.imshow(xyz_to_srgb(img), vmin = 0, vmax = 1)
		plt.savefig(img_save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
		plt.close()
		# ax00.set_xticks([])
		# ax00.set_yticks([])
		# ax00.imshow(xyz_to_srgb(img), vmin = 0, vmax = 1)
		# ax00.set_title(f"Image: {fn}", fontsize='x-large')

		plt.figure()
		plt.xticks([])
		plt.yticks([])
		# ax01.set_xticks([])
		# ax01.set_yticks([])
		color = np.ones_like(img)
		H, W, C = img.shape
		Y_l = np.linspace(0.125, 1.0, slots)
		for i, Y in enumerate(Y_l):
			xyY = np.hstack((prop['xy'], Y))
			XYZ = xyY2xyz(xyY)
			offset = W // slots
			color[:, i * offset : (i+1) * offset] = color[:, i * offset : (i+1) * offset] * XYZ

		color = xyz_to_srgb(color) # converts to sRGB with gamma 2.2
		plt.imshow(color)
		# ax01.set_title(f"Color sRGB", fontsize='x-large')

		# Y = get_Y(img)
		# ax11.set_xticks([])
		# ax11.set_yticks([])
		# idx = prop['argmaxY']
		# maxY = prop['maxY']
		# ax11.plot(idx[1], idx[0], marker="*", markersize=4, alpha=.5, markeredgecolor="black", markerfacecolor="black", label=f"max: {round(maxY,2)}")
		# ax11.legend()
		# Yref = ax11.imshow(Y)
		# plt.colorbar(Yref, shrink=.5, ticks=np.round(np.linspace(0,1, num=6),2).tolist())
		# ax11.set_title("Luminocity", fontsize='x-large')

		plt.savefig(col_save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
		plt.close()


def fetch_raw_paths(dir_path):
	paths = []
	for root, dirs, files in os.walk(dir_path, topdown=False):
		for name in files:
			if not name.endswith('.dng'):
				print(f'file: {name} is not a .DNG file. Skipping...')
				continue
			paths.append(os.path.join(root, name))

	return paths


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('img_dir', type=str, help='Path to raw DNG files. (Convert from CR2 or NEF to DNG using AdobeDNG converter.)')
	arg_parser.add_argument('out_dir', type=str, help='The output directory.')
	arg_parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='Directory to save a checkpoint of the annotation phase.')
	arg_parser.add_argument('--load_ckpt', action='store_true', default=False, help='Load from checkpoint.')
	args = arg_parser.parse_args()

	img_dir = args.img_dir
	ckpt_dir = args.ckpt_dir
	load_ckpt = args.load_ckpt
	os.makedirs(ckpt_dir, exist_ok=True)

	output_dir = args.out_dir
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, 'data.json')
	
	paths = fetch_raw_paths(img_dir)
	names = [os.path.basename(p) for p in paths]

	if not load_ckpt:
		process_images(paths, ckpt_dir)

	# compute image properties.
	props_l = compute_props(ckpt_dir)

	# dump all to a csv.
	print(f'Writing properties to: {output_path}')
	with open(output_path, 'w', encoding='utf-8') as fp:
		json.dump(props_l, fp, ensure_ascii=False, indent=4, cls=NumpyEncoder)

	plot_chromaticities(props_l, ckpt_dir, output_dir, dpi=800)