"""
	Post-processing pipeline:
	1. auto_white_balance
	2. srgb
	3. gamma
	4. tone
	5. local_tone_mapping
	6. gaussian noise

"""
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing
import cv2

import tifffile as tiff
from skimage.color import xyz2lab, lab2xyz

from noise_profiler.image_synthesizer import load_noise_model
from camera_pipeline import xyz_to_srgb


def remap(x, omin, omax, nmin, nmax):
	x = (x - omin) / (omax - omin) * (nmax - nmin) + nmin
	return np.clip(x, a_min=nmin, a_max=nmax)


def save_png_mask(path:str, m:np.ndarray, dtype='uint16'):
	""" 
		Args:
				path (str): path to store the data.
				m (np.ndarray): should be in range [0,1].
				dtype (str): image data type (uint8, uint16).
	"""
	a_max = np.max(m)
	a_min = np.min(m)
	if a_max > 1 or a_min < 0:
		# print('Warning: not in the nominal range [0,1], remapping enforced.')
		m = remap(m, a_min, a_max, 0, 1)


	has_color = lambda x: (len(x.shape) == 3)
	img_t = dtype

	m = m * np.iinfo(img_t).max

	if m.dtype != img_t:
		m = m.astype(img_t)

	if has_color(m):
		m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)

	cv2.imwrite(path + '.png', m, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def load_tiff_mask(path:str):
	return tiff.imread(path)


def apply_pipeline(img, op_list):
	result_l = []
	for op in op_list:
		img = globals()[op](img)
		result_l.append(img.copy())

	return result_l


def auto_white_balance(img_xyz):
	"""
		Using gray world assumption for illuminant estimation.
		Args:
				img_xyz (np.ndarray): XYZ image in nominal range [0,1].
	"""
	H, W, C = img_xyz.shape
	
	# WP estimation.
	# the conversion to Lab does not requires the white point, but a reference point.
	img_lab = xyz2lab(img_xyz)
	illum_lab = np.mean(img_lab, axis=(0,1))
	as_shot_neutral = lab2xyz(illum_lab)
	
	for i in range(C) :
		img_xyz[..., i] = img_xyz[..., i] / as_shot_neutral[i] * as_shot_neutral[1]
	
	return  np.clip(img_xyz, 0.0, 1.0)


noise_model_path = 'noise_profiler/h-gauss-s20-v1'
noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_noise_model(path=noise_model_path)
def gaussian_noise(src_image, model=noise_model, dst_iso=3200, spat_var=False, iso2b1_interp_splines=iso2b1_interp_splines , iso2b2_interp_splines=iso2b2_interp_splines):
	"""
		Synthesize a noisy image from `src_image` using a heteroscedastic Gaussian noise model `model`.
		:param src_image: Clean/Semi-clean image.
		:param model: Noise model.
		:param dst_iso: ISO of noisy image to be synthesized.
		:param spat_var: Simulate spatial variations in noise.
		:param iso2b1_interp_splines: Interpolation/extrapolation splines for shot noise (beta1)
		:param iso2b2_interp_splines: Interpolation/extrapolation splines for read noise (beta2)
	"""
	# make a copy
	image = src_image.copy().astype(np.float32)
	image = image * 65535

	# if target ISO is not specified, select a random value
	if dst_iso is None:
		dst_iso = np.random.randint(50, 3201)

	if iso2b1_interp_splines is None or iso2b2_interp_splines is None:
		iso2b1_interp_splines = model['iso2b1_interp_splines']
		iso2b2_interp_splines = model['iso2b2_interp_splines']

	# get noise params (shot, read), per channel
	if dst_iso in model:
		dst_params = model[dst_iso]
	else:
		dst_params = np.zeros((4, 2))
		for c in range(4):
			dst_params[c, 0] = iso2b1_interp_splines[c](dst_iso)
			dst_params[c, 1] = iso2b2_interp_splines[c](dst_iso)

	# compute noise variance, std. dev.
	dst_var = np.zeros(shape=image.shape)
	#                                             β1                β2
	dst_var[...,0] = image[...,0] * dst_params[1][0] + dst_params[1][1]
	dst_var[...,1] = image[...,1] * dst_params[0][0] + dst_params[0][1]
	dst_var[...,2] = image[...,2] * dst_params[2][0] + dst_params[2][1]

	# simulate variance of noise variance
	if spat_var:
		dst_var[dst_var < 0] = 0
		dst_var += np.random.normal(loc=0, scale=1, size=image.shape) * np.sqrt(dst_var)

	dst_var[dst_var < 0] = 0

	# Normal Gaussian noise
	# scaled by heteroscedastic standard deviation
	noise = np.random.normal(loc=0, scale=1, size=image.shape)
	#noise *= np.sqrt(dst_var) * 7
	noise *= np.sqrt(dst_var) * 3

	# add noise
	noisy_image = image + noise
	noisy_image = noisy_image / 65535
	noisy_image = np.clip(noisy_image, a_min=0, a_max=1)

	return noisy_image


def normalize(img):
	# input image has 16-bit color.
	# BUG: expecting uint16
	return remap(img, omin=0, omax=65535, nmin=0, nmax=1)

def exposure(img):
	# to prevent overflow.
	img = np.double(img)

	exp = 3.25 # SOLOv3
	img = img * (2 ** exp)
	img = np.clip(img, 0, 65535).astype('uint16')
	return img

op_list = [ 'exposure', 'normalize' ] # gets overwritten.

def store_results(result_l, fn, out_dir, all_res=False):

	dtype = 'uint16'
	if not all_res:
		if op_list[-1] == 'gaussian_noise':
			dtype = 'uint8'

		out_path = os.path.join(out_dir, fn.split('.')[0])
		save_png_mask(out_path, result_l[-1], dtype)
		return

	for i, res in enumerate(result_l):
		os.makedirs(os.path.join(out_dir, op_list[i]), exist_ok=True)
		name = '_'.join((fn.split('.')[0], op_list[i]))

		if op_list[i] == 'gaussian_noise':
			dtype = 'uint8'

		out_path = os.path.join(out_dir, op_list[i], name)
		save_png_mask(out_path, res, dtype)


def worker(proc_id, fns):

	if proc_id == 0:
		sequence = tqdm(fns, total=len(fns), dynamic_ncols=True)
	else:
		sequence = fns

	for fn in sequence:
		img = load_tiff_mask(os.path.join(img_dir, fn))
		result_l = apply_pipeline(img, op_list)
		store_results(result_l, fn, out_dir, all_res=store_all)


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('--img_dir', type=str, help='Linear XYZ image directory path.')
	arg_parser.add_argument('--out_dir', type=str, default='.', help='Output directory path.')
	arg_parser.add_argument('--img_path', type=str, help='Use to process a single image.')
	arg_parser.add_argument('--store_all', action='store_true', default=False, help='To store all the intermediate results.')
	arg_parser.add_argument('--workers', type=int, default=16, help='Number of workers to use.')
	arg_parser.add_argument('--op_list', action='store', type=str, nargs='*', help='Output directory path.')
	args = arg_parser.parse_args()
	
	img_dir = args.img_dir
	out_dir = args.out_dir
	store_all = args.store_all
	img_path = args.img_path
	workers = args.workers

	if args.op_list is not None and len(args.op_list) != 0:
		op_list = args.op_list

	os.makedirs(out_dir, exist_ok=True)

	if img_path is not None:
		img = load_tiff_mask(img_path)
		result_l = apply_pipeline(img, op_list)
		store_results(result_l, os.path.basename(img_path), out_dir, all_res=store_all)

	else:

		file_names = os.listdir(img_dir)

		cores = len(os.sched_getaffinity(0))
		if workers > cores:
			workers = cores

		if workers == 1:
			proc_id = 0
			worker(proc_id, file_names)
			exit(0)

		file_names_core = np.array_split(list(file_names), cores)

		print("Number of cores: {}, images per core: {}".format(cores, len(file_names_core[0])))
		workers = multiprocessing.Pool(processes=cores)
		processes = []
		for proc_id, fns in enumerate(file_names_core):
			p = workers.apply_async(worker, ( proc_id, fns))
			processes.append(p)

		for p in processes:
			p.wait()

