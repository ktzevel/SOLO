"""
	Adjusted from: https://github.com/cocodataset/panopticapi.git
"""

import json
import os
import cv2
import numpy as np
import PIL.Image as Image
from pycocotools import mask as COCOmask
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage 

from collections.abc import Iterable

class IdGenerator():
	'''
	The class is designed to generate unique IDs that have meaningful RGB encoding.
	Given semantic category unique ID will be generated and its RGB encoding will
	have color close to the predefined semantic category color.
	The RGB encoding used is ID = R + 256 * G + 256**2 * B.
	Class constructor takes dictionary {id: category_info}, where all semantic
	class ids are presented and category_info record is a dict with fields
	'isthing' and 'color'
	'''
	def __init__(self, categories:dict, segments_info=None):
		
		self.taken_colors = set([0, 0, 0])
		if segments_info is not None:
			self.taken_colors.update([tuple(id2rgb(s['id'])) for s in segments_info])
		
		self.categories = categories
		for category in self.categories.values():
			if category['isthing'] == 0:
				self.taken_colors.add(tuple(category['color']))

	def get_color(self, cat_id):
		def random_color(base, max_dist=30):
			new_color = base + np.random.randint(low=-max_dist,
												 high=max_dist+1,
												 size=3)
			return tuple(np.maximum(0, np.minimum(255, new_color)))

		category = self.categories[cat_id]
		if category['isthing'] == 0:
			return category['color']
		base_color_array = category['color']
		base_color = tuple(base_color_array)
		if base_color not in self.taken_colors:
			self.taken_colors.add(base_color)
			return base_color
		else:
			while True:
				color = random_color(base_color_array)
				if color not in self.taken_colors:
					self.taken_colors.add(color)
					return color
	
	def get_id(self, cat_id):
		color = self.get_color(cat_id)
		return rgb2id(color)

	def get_id_and_color(self, cat_id):
		color = self.get_color(cat_id)
		return rgb2id(color), color

class Encoder(json.JSONEncoder):

	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)

		if isinstance(obj, np.floating):
			return float(obj)

		if isinstance(obj, np.ndarray):
			return obj.tolist()

		return super(Encoder, self).default(obj)

def rgb2id(color):
	if isinstance(color, np.ndarray) and len(color.shape) == 3:
		if color.dtype == np.uint8:
			color = color.astype(np.int32)
		return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
	return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def id2rgb(id_map):
	if isinstance(id_map, np.ndarray):
		id_map_copy = id_map.copy()
		rgb_shape = tuple(list(id_map.shape) + [3])
		rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
		for i in range(3):
			rgb_map[..., i] = id_map_copy % 256
			id_map_copy //= 256
		return rgb_map
	color = []
	for _ in range(3):
		color.append(id_map % 256)
		id_map //= 256
	return color

def rgb2sc(img:np.ndarray):
	"""
		Converts an RGB image to a single channel image, using the formula:
		ID = R + 256*G + 256^2*B
	"""
	h, w, _ = img.shape
	ones = np.ones((h, w))
	weights = np.transpose(np.stack((ones, ones*256, ones*256**2)), (1,2,0))
	prod = img * weights
	return np.sum(prod, axis=2)

def merge(m1, m2, intersect):
	''' Returns a mask in RLE format. '''
	masks = np.asfortranarray(np.stack((m1, m2), axis=-1))
	Rs = COCOmask.encode(masks)
	return COCOmask.merge(Rs, intersect)

def union_mask(m1, m2)->np.ndarray:
	R = merge(m1, m2, intersect=False)
	return 255 * COCOmask.decode(R)

def union_area(m1, m2)->int:
	R = merge(m1, m2, intersect=False)
	return COCOmask.area(R)

def intersection_mask(m1, m2)->np.ndarray:
	R = merge(m1, m2, intersect=True)
	return 255 * COCOmask.decode(R)

def intersection_area(m1, m2)->int:
	R = merge(m1, m2, intersect=True)
	return COCOmask.area(R)

def mask2area(m)->float:
	R = COCOmask.encode(np.asfortranarray(m))
	return COCOmask.area(R)

def load_mask(path):
	return np.array(Image.open(path))

def to_binary_mask(rgb_m, val):
	
	if isinstance(val, (list, tuple)):
		val = np.array(val) # val is an rgb triplet.
	else:
		val = int(val)

	bm = cv2.inRange(rgb_m, val, val)
	return bm

def boundary_mask(shape):
	h, w = shape
	pad_size = 1

	h = int(h-2*pad_size)
	w = int(w-2*pad_size)

	pad_val = 255
	return np.pad(np.zeros((h, w), dtype='uint8'), pad_size, constant_values=(pad_val,))

def mask2bbox(m):
	Rs = COCOmask.encode(np.asfortranarray(m))
	x, y, w, h = [ int(e) for e in COCOmask.toBbox(Rs)]
	return x, y, w, h

def bbox2mask(x, y, w, h, shape):
	m = np.zeros(shape, dtype='uint8')
	m[y:y+h, x:x+w] = 255
	return m


def get_fid(filename):
	return ('_'.join(filename.split('_')[:3]).split('.')[0])

def save_json(d, file):
	with open(file, 'w') as f:
		json.dump(d, f, cls=Encoder)

def resize_move(f, w, h, x, y):
	backend = matplotlib.get_backend()
	if backend == 'TkAgg':
		f.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w, h, x, y))

def plot_grid(*arrays, w=1920, h=1080, x=0, y=0):
	arrays = list(arrays)
	no = len(arrays)

	f = plt.figure()
	resize_move(f, w, h, x, y)
	gs = f.add_gridspec(1, no, hspace=0, wspace=0)
	gs.update(left=0,right=1,top=1,bottom=0)
	axes = gs.subplots()
	for i, ax in enumerate(axes):
		ax.imshow(arrays[i])
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()

def plot_overlayed(img, ls, acdc, w=1920, h=1080, x=1920, y=0):
	
	# arrays = list(arrays)
	# no = len(arrays)

	f, ax = plt.subplots()
	ax.set_xticks([])
	ax.set_yticks([])

	resize_move(f, w, h, x, y)

	plt.imshow(img, alpha=.4)
	plt.show(block=False)
	plt.imshow(ls, alpha=.9)
	plt.show(block=False)
	plt.imshow(acdc, alpha=.4)
	plt.show(block=False)

	plt.show()

def plot_custom(img, ls, acdc, lm, save_path, dpi):

	f = plt.figure()
	f.canvas.manager.full_screen_toggle()

	gs = f.add_gridspec(1, 2, hspace=0, wspace=0)
	gs.update(left=0,right=1,top=1,bottom=0)

	axes = gs.subplots()
	if not isinstance(axes, Iterable):
		axes = [axes]
	
	for ax in axes:
		ax.set_xticks([])
		ax.set_yticks([])

	f.suptitle('Light Sources', fontsize='small')
	axes[0].imshow(img, alpha=1)
	axes[0].imshow(acdc, alpha=.4)
	axes[0].imshow(ls, alpha=.7)
	axes[0].set_title('Annotated', fontsize='small')

	axes[1].imshow(img, alpha=1)

	lm = np.repeat(lm[..., None],3, axis=-1).astype('uint8')
	axes[1].imshow(lm, alpha=.7)
	axes[1].set_title('Activated', fontsize='small')

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, dpi=dpi)