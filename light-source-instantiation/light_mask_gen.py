import os
import json
import itertools
import argparse
from tqdm import tqdm
import numpy as np
import PIL.Image as Image
from scipy import ndimage 
from skimage.color import xyz2rgb
import cv2
import tifffile as tiff

import labels
from utils import *

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

class Dataset:
	def __init__(self, path, name):
		print(path)
		assert(os.path.isdir(path))
		self.dir = path

		self.name = name
		self.data = self._load_json()
		self.categories = {cat['id']: cat for cat in self.data['categories']}
		self.images = self.data['images']
		self.annotations = self.data['annotations']
		
		img = self.data['images'][0]
		self.image_shape = [img['height'], img['width']]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, key):
		return self.data[key]

	def _load_json(self):
		fns = os.listdir(self.dir)
		l = [fn for fn in fns if fn.endswith('.json')]
		if not l or len(l) > 1:
			raise RuntimeError(f'The file structure is not compatible.')
		
		json_fn = l[0]
		with open(os.path.join(self.dir, json_fn)) as fp:
			return json.load(fp)

class AnnotationsIterator:
	def __init__(self, *args:Dataset):
		self.ds = args
		self.fn2id_maps = []
		for d in self.ds:
			self.fn2id_maps.append({img['file_name']:img['id'] for img in d['images']})

		lens = [len(m) for m in self.fn2id_maps]
		min_idx = lens.index(min(lens))
		self.fn_list = list(self.fn2id_maps[min_idx].keys())
		self.max = len(self.fn_list)

	def __iter__(self):
		self.idx = 0
		return self

	def __next__(self):
		if self.idx < self.max:
			fn = self.fn_list[self.idx]
			
			if get_fid(fn) in labels.fn_id_exclusion_list:
				self.idx += 1
				return self.__next__()

			annotations = []
			for i, d in enumerate(self.ds):
				img_id = self.fn2id_maps[i][fn]
				annotations.append(self._get_ann(d['annotations'], img_id))

			self.idx += 1
			return annotations
		else:
			raise StopIteration
	
	def __len__(self):
		return self.max - len(labels.fn_id_exclusion_list)
	
	def _get_ann(self, ann_l, img_id):
		for item in ann_l:
			if item['image_id'] == img_id:
				return item
		raise RuntimeError(f'Annotation with image_id:{img_id} was not found.')

class LightTree:

	masks_out_dir = ''

	def __init__(self, img_fn_id, img_shape, cat2xyz, depth_dir, seed=None):
		np.random.seed(seed=seed)
		self.root = {
			  'name':'root'
			, 'mask': None
			, 'children':[]
			, 'parents':[]
			}

		self.fn_id = img_fn_id
		self.cat2xyz = cat2xyz
		self.img_shape = img_shape

		self.strength_mask_dict = {}
		self.street_lights = []
		H, W = self.img_shape
		self.xyz_mask = np.zeros((H, W, 3), dtype=np.double)

		fid2depth = {}
		for root, _, files in os.walk(depth_dir):
			for file in files:
				if not file.endswith('.npy'):
					continue
				
				fid = file.split('.')[0]
				fid2depth[fid]= os.path.join(root, file)

		self.fid2depth = fid2depth
	
	def build(self, seg_l):

		def add_to_root(*nodes):
			nodes = list(nodes)
			self.root['children'].extend(nodes)
		
		def compatible(sn, gn):
			return gn in labels.source_groups[sn]
		
		def trim_masks(node):
			if node['mask'] is None:
				return # root node case.

			for c in node['children']:
				if 'children' in c:
					continue

				c['mask'] = intersection_mask(c['mask'], node['mask'])

		def split_lights(node, light_cat):
			children = []
			processed = []
			for i, c in enumerate(node['children']):
				cn = c['name']

				if cn not in labels.__dict__[light_cat]:
					continue

				processed.append(i)
				lbl_mask, n = ndimage.label(c['mask'])
				if n > 2:
					print(f'Found {n} segmentations for {light_cat}.')

				masks = []
				for l in range(1, n+1):
					masks.append(255 * (lbl_mask==l).astype('uint8'))
				
				for m in masks:
					children.append({
						  'name': cn
						, 'mask': m
						, 'parents': c['parents']
					})

			# drops processed nodes from the node's children.
			num = len(node['children'])
			for i in range(num):
				if i in processed:
					continue
				children.append(node['children'][i])

			node['children'] = children

		groups = []
		sources = []
		for s in seg_l:
			(sources, groups)[s['name'] in labels.groups].append(s)

		for g in groups:
			g['parents'] = ['root']

		for s in sources:
			area_l = []
			idx_l = []
			for gi, g in enumerate(groups):
				if not compatible(s['name'], g['name']):
					continue

				area_l.append(intersection_area(s['mask'], g['mask']))
				idx_l.append(gi)
			
			max_area = max(area_l) if area_l else 0
			if max_area <= 0:
				s['parents'] = ['root']
				add_to_root(s)
				continue # orphan nodes.

			gi = idx_l[area_l.index(max_area)]
			if 'children' not in groups[gi]:
				groups[gi]['children'] = []
			
			s['parents'] = groups[gi]['parents'].copy()
			s['parents'].append(groups[gi]['name'])
			groups[gi]['children'].append(s)
		
		add_to_root(*groups)
		self._walk(self.root, trim_masks)
		self._walk(self.root, split_lights, light_cat='head_lights')
		self._walk(self.root, split_lights, light_cat='street_lights')
	
	def generate_light_mask(self):

		def set_states(node):
			gn = node['name']
			name2params = {}
			for c in node['children']:
				# for each child of the group node.

				if 'children' in c:
 					# child is a group node.
					continue
				
				# child is a leaf node (group or source).
				cn = c['name']
				if cn in labels.groups:
					print(f'{cn} group has no children, skipping...')
					continue

				# child is a light source.
				if cn not in name2params.keys() or gn == 'root':
					# first time this light source type (=cn) is observed for the current group node.
					
					key_l = c['parents'] + [cn] # the name path from the root node to the light source node.

					params = labels.get_uniform_params(*key_l)
					if params is None:
						# the name path (key_l) provided does't correspond to a set of bernouli params.
						# this node is located in a wrong hierarchy.
						print(f'orphan {cn} in {c["parents"]}, skipping...')
						continue
	 
					if cn not in cat2xyz:
						# there are some light sources that we have no samples for.
						# e.g. advertisements.
						continue

					# init. a params dict.
					name2params[cn] = {}

					# bernoulli p:
					low, high = params
					name2params[cn]['bernoulli_p'] =  np.random.uniform(low, high)

					# chroma xyz:
					sample_size = cat2xyz[cn].shape[0]
					i = np.random.randint(low=0, high=sample_size, size=1)[0]
					name2params[cn]['xyz'] = cat2xyz[cn][i]

					# strength:
					low, high = labels.strength[cn]
					name2params[cn]['strength'] = np.random.uniform(low, high, size=1)


				p = name2params[cn]['bernoulli_p']
				c['_p'] = np.round(p, decimals=2) # for debug.
				c['is_on'] = bool(np.random.binomial(n=1, p=p))
				c['xyz'] = name2params[cn]['xyz']
				c['strength'] = name2params[cn]['strength']
		
		def create_mask(node):
			for c in node['children']:
				if 'children' in c:
					continue

				cn = c['name']

				if c.get('is_on'):
					# c is an active light source
					m = (c['mask'] == 255)

					if c['name'] not in labels.street_lights:

						self.xyz_mask[m] = c['xyz']
						
						# strength should be kept per category for blender to assign the proper directionality.
						if cn not in self.strength_mask_dict.keys():
							self.strength_mask_dict[cn] = np.zeros(self.img_shape, dtype=np.single)

						self.strength_mask_dict[cn][m] = c['strength']
					
					else:
						depth = np.load(self.fid2depth[self.fn_id])
						
						# get the coordinates of the street lights
						# calc. the middle point at the bottom of the annotation.
						offset = 1.5 # px, a small offset is required in y-axis to position the light source under the mesh part which corresponds to the street light.
						mass_i, mass_j = np.where(m)
						ys = int(np.max(mass_i)) + offset
						xs = int(np.average(mass_j))


						# calc. the average depth of the mask of the street light.
						z = np.average(depth[m])

						# TODO: get parameters from the calibration file.
						fx = 1780
						fy = 1780
						cx = 959.5
						cy = 539.5

						x = (xs - cx) / fx
						y = (ys - cy) / fy

						x = x * z
						y = y * z
						
						# for the side size of the artificial street light.
						ys_delta = np.ptp(mass_i)
						xs_delta = np.ptp(mass_j)

						# calc. side length based on a delta average.
						sz_upper_lim = 1.5 # m, size limit for unaccounted cases.
						sz_lower_lim = 0.30 # m, size limit for unaccounted cases.
						size =  (ys_delta * z / fy + xs_delta * z / fx) / 2
						if size > sz_upper_lim:
							size = sz_upper_lim
						elif size < sz_lower_lim:
							size = sz_lower_lim

						coords = np.array([x, y, z])
						if np.isnan(coords).any():
							raise RuntimeError('The coordinates of the street lights can\'t be NaN')

						self.street_lights.append({
							    'coords': coords
							  , 'strength': c['strength'][0]
							  , 'rgb': xyz2rgb(c['xyz'])
							  , 'size': size
							}
						)

		def store_masks():
			base_path = os.path.join(self.masks_out_dir, self.fn_id)
			strength_dir = os.path.join(base_path, 'strength')
			os.makedirs(strength_dir, exist_ok=True)

			# store chromaticity mask:
			xyz_path = os.path.join(base_path, 'xyz.png')
			save_xyz_mask(xyz_path, self.xyz_mask)

			# store strength masks:
			for cat, mask in self.strength_mask_dict.items():
				str_path = os.path.join(strength_dir, f'{cat}.tiff')
				save_tiff_mask(str_path, mask)
			
			# store street lights info.:
			path = os.path.join(base_path, 'street_lights.json')
			with open(path, 'w', encoding='utf-8') as fp:
				json.dump(self.street_lights, fp, ensure_ascii=False, indent=4, cls=NumpyEncoder)

		self._walk(self.root, set_states)
		self._walk(self.root, create_mask)
		store_masks()

	def _walk(self, node, visit, **kargs):
		if not node.get('children'): return
		visit(node, **kargs)
		for c in node['children']:
			self._walk(c, visit, **kargs)
	

def remap(x, omin, omax, nmin, nmax):
	return (x - omin) / (omax - omin) * (nmax - nmin) + nmin


def save_tiff_mask(path:str, m:np.ndarray):
	tiff.imwrite(path, m)


def save_xyz_mask(path:str, m:np.ndarray):
	has_color = lambda x: (len(x.shape) == 3)
	img_t = 'uint16'

	m = remap(m, omin=0, omax=1, nmin=0, nmax=np.iinfo(img_t).max)

	m = m.astype(img_t)

	if has_color(m):
		m = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)

	cv2.imwrite(path, m, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def get_cat2xyz(json_path):
	cat2xyz = {}
	light_l = []
	with open(json_path, 'r') as fp:
		light_l = json.load(fp)

	for light in light_l:
		cat = light['cat']
		XYZ = light['XYZ_no_illum']
		if cat not in cat2xyz:
			cat2xyz[cat] = [XYZ]
			continue

		cat2xyz[cat].append(XYZ)
		
	for k in cat2xyz.keys():
		cat2xyz[k] = np.array(cat2xyz[k])

	return cat2xyz

#def export_semantic_masks(seg_l, out_dir, fn_id):
#
#	if not seg_l:
#		return
#
#	out_path = os.path.join(out_dir, fn_id, 'materials')
#	os.makedirs(out_path, exist_ok=True)
#
#	metal_m = np.zeros_like(seg_l[0]['mask'])
#	glass_m = np.zeros_like(seg_l[0]['mask'])
#	asphalt_m = np.zeros_like(seg_l[0]['mask'])
#
#	for s in seg_l:
#		m = (s['mask'] == 255)
#
#		if s['name'] in labels.metal:
#			metal_m[m] = s['mask'][m]
#		elif s['name'] in labels.glass:
#			glass_m[m] = s['mask'][m]
#		elif s['name'] in labels.asphalt:
#			asphalt_m[m] = s['mask'][m]
#
#	# masks should be disjoint:
#	# vehicle annotations include their lights.
#	metal_m = np.abs(metal_m - glass_m)
#
#	# store masks
#	metal_path = os.path.join(out_path, 'metal.tiff')
#	glass_path = os.path.join(out_path, 'glass.tiff')
#	asphalt_path = os.path.join(out_path, 'asphalt.tiff')
#
#	save_tiff_mask(metal_path, metal_m)
#	save_tiff_mask(glass_path, glass_m)
#	save_tiff_mask(asphalt_path, asphalt_m)


#def remove_unnecessary_segs(seg_l):
#	new_l = []
#	for s in seg_l:
#		if s['name'] not in labels.ls.names:
#			continue
#		new_l.append(s)
#
#	return new_l

def conform(annotation, dname, ddir):
	"""
		Performs the following tasks:
		- converts annotations category_ids to the corresponding LS category names.
		- for each annotation only the name of the category and the binary mask are kept.
		- from ACDC's annotations only the vehicles annotations are preserved.
		- fills the holes in the LS windows_group masks so that the intersection can be computed later on.
	"""

	if dname not in ['acdc', 'ls']:
		raise RuntimeError(f"Dataset name: {dname} is not supported.")

	seg_l = []
	for s in annotation['segments_info']:

		name = labels.__dict__[dname].id2name[s['category_id']]
		if dname == 'acdc':
			# for ACDC filter all non-vehicle segments.
			if name not in labels.vehicles:
				continue

		fn = annotation['file_name']
		path = os.path.join(ddir, 'coco-panoptic', fn)
		m = to_binary_mask(load_mask(path), id2rgb(s['id']))

		if name == 'windows_group':
			# handles the case where the windows_group mask is open due to the image boundaries.
			bm = boundary_mask(img_shape)
			bbm = bbox2mask(*mask2bbox(m), m.shape)
			im = intersection_mask(bbm, bm)
			m = union_mask(m, im)

			# fills the window holes in the windows_group mask.
			m = 255 * ndimage.binary_fill_holes(m).astype('uint8')

		seg_l.append({'name': name, 'mask': m})

	return seg_l

def rm_duplicates(seg_l, ls_idx, acdc_idx):
	"""
		Removes vehicle annotations in ACDC that have been overwritten in LS.
		Args:
			seg_l (list): a list of segmentations lists i.e. one for each dataset.
			ls_idx (int): the index of LS dataset in seg_l.
			acdc_idx (int): the index of ACDC dataset in seg_l.
	"""

	ls_segs = seg_l[ls_idx]
	acdc_segs = seg_l[acdc_idx]

	for ls_s in ls_segs:
		if ls_s['name'] in labels.vehicles:
			inter_l = []
			idx_l = []
			for i, acdc_s in enumerate(acdc_segs):
				if acdc_s['name'] in labels.vehicles:
					inter = intersection_area(ls_s['mask'], acdc_s['mask'])
					if inter > 0:
						idx_l.append(i)
						inter_l.append(inter/mask2area(acdc_s['mask']))

			if inter_l:
				max_inter_idx = idx_l[inter_l.index(max(inter_l))]
				del acdc_segs[max_inter_idx]
	
	return seg_l

"""
	python light_mask_gen.py --ls=data/LS_ALL_PAN/ --acdc=data/ACDC_ALL_PAN --xyz=data/XYZ.json --depth=data/unidepth-conv-bilinear -o=light_masks_1432 -s=1432
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Probabilistic light source module.")
	parser.add_argument('--acdc', type=str, help="ACDC dataset's path.", required=True)
	parser.add_argument('--ls', type=str, help="Light sources dataset's path.", required=True)
	parser.add_argument('--xyz', type=str, help="Path to the XYZ .json file", required=True)
	parser.add_argument('--depth', type=str, help="Path to the refined depth dir.", required=True) # get that from the reconstruction code.
	parser.add_argument('-o', '--out_dir', type=str, help="the directory to generate light masks.", required=True)
	parser.add_argument('-s', '--seed', type=int, help="the seed for the random generator.", default=None)
	parser.add_argument('--fids', type=str, help="the path to a file containing a list of fids to be processed.")
	args = parser.parse_args()

	ds = [Dataset(args.ls, "ls"), Dataset(args.acdc, "acdc")]
	out_dir = args.out_dir
	json_path = args.xyz
	seed = args.seed
	depth_dir = args.depth
	fids_path = args.fids

	LightTree.masks_out_dir = out_dir
	os.makedirs(out_dir, exist_ok=True)
	
	cat2xyz = get_cat2xyz(json_path)

	fids = []
	if fids_path:
		with open(fids_path, 'r') as fp:
			fids = fp.read().splitlines()

	for ann_l in tqdm(AnnotationsIterator(*ds), dynamic_ncols=True):

		ann_fn = ann_l[0]['file_name']
		fn_id = get_fid(ann_fn)

		if fids and fn_id not in fids:
			continue
		
		img_shape = ds[0].image_shape

		ds_seg_l = []
		ds_no = len(ds)
		for i in range(ds_no):
			ds_seg_l.append(
				conform(ann_l[i], ds[i].name, ds[i].dir)
			)

		names = [d.name for d in ds]
		ds_seg_l = rm_duplicates(ds_seg_l, names.index('ls'), names.index('acdc'))
		seg_l = list(itertools.chain(*ds_seg_l))

		#export_semantic_masks(seg_l, out_dir, fn_id)
		#seg_l = remove_unnecessary_segs(seg_l)

		tree = LightTree(fn_id, img_shape, cat2xyz, depth_dir, seed=seed)
		tree.build(seg_l)
		tree.generate_light_mask()
