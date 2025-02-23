"""
	This script combines the semantic and instance annotations of the ACDC reference images into the panoptic format.

	Usage examples:
		python acdc2panoptic.py -sACDC_GT/data/semantic -iACDC_GT/data/instance -oACDC_PAN

		python acdc2panoptic.py -s=data/LS_ALL/semantic/all -i=data/LS_ALL/instance/all -o=LS_ALL_PAN --label_set=ls
"""

import os
import copy
import time
import argparse
from tqdm import tqdm
import multiprocessing

import numpy as np
import cv2
import PIL.Image as Image

from utils import *
import labels

def _to_int_list(array:np.ndarray):
	return [int(e) for e in array.tolist()]

def main_worker(sem_dir, inst_dir, json_file, ann_dir, label_set, single_worker=False):

	start_time = time.time()

	def num_comp(fn):
		return int(''.join(list(filter(str.isdigit, fn))))

	ann_paths = []
	for d in (sem_dir, inst_dir):
		name_list = sorted([fn for fn in os.listdir(d)], key=num_comp)
		path_list = [os.path.join(d, fn) for fn in name_list]
		ann_paths.append(path_list)
	
	fids = sorted([get_fid(f) for f in os.listdir(sem_dir)], key=num_comp)

	img_names = [f + '.png' for f in fids]
	ann_names = img_names.copy()

	names_and_paths = []
	items_no = len(ann_paths[0])
	for i in range(items_no):
		names_and_paths.append(
			{
				'img_name': img_names[i],
				'ann_name': ann_names[i],
				'sem_path': ann_paths[0][i],
				'inst_path': ann_paths[1][i]
			}
		)

	coco_d = {
		'info': {
			'description': 'ACDC panoptic annotations for reference images.',
			'version': 'v0.1'
		},
		'categories': [],
		'images': [],
		'annotations': []
	}

	for l in labels.__dict__[label_set].labels:
		coco_d['categories'].append(
			{
				"id": l.id,
				"name": l.name,
				"color": l.color,
				"attributes": None,
				"isthing": int(l.hasInstances)
			}
		)
	
	for i, item in enumerate(names_and_paths):
		coco_d['images'].append(
			{
				"id": i,
				"file_name": item['img_name'],
				"height": 1080,
				"width": 1920
			}
		)

	img_ids = [img['id'] for img in coco_d['images']]

	if single_worker:
		proc_id = 0
		coco_d = worker(proc_id, img_ids, coco_d, names_and_paths, ann_dir)
		save_json(coco_d, json_file)
		return
	
	cpu_num = len(os.sched_getaffinity(0))
	img_ids_split = np.array_split(list(img_ids), cpu_num)

	print("Number of cpus: {}, images per cpu: {}".format(cpu_num, len(img_ids_split[0])))
	workers = multiprocessing.Pool(processes=cpu_num)
	processes = []
	for proc_id, img_ids in enumerate(img_ids_split):
		p = workers.apply_async(worker, ( proc_id,
										  img_ids,
										  copy.deepcopy(coco_d),
										  names_and_paths,
										  ann_dir
										))
		processes.append(p)

	annotations = []
	for p in processes:
		partial_coco_d = p.get()
		annotations.extend(partial_coco_d['annotations'])
	
	# sort based on image_id.
	annotations = sorted(annotations, key=lambda ann:ann['image_id'])

	coco_d['annotations'] = annotations
	save_json(coco_d, json_file)

	t_delta = time.time() - start_time
	print("Time elapsed: {:0.2f} minutes".format(t_delta/60))

def worker(proc_id, img_ids, coco_d, names_and_paths, ann_dir):

	if proc_id == 0:
		sequence = tqdm(img_ids, total=len(img_ids), dynamic_ncols=True)
	else:
		sequence = img_ids

	for im_id in sequence:
		id_gen = IdGenerator({cat['id']: cat for cat in coco_d['categories']})

		sem_mask = load_mask(names_and_paths[im_id]['sem_path'])
		inst_mask = load_mask(names_and_paths[im_id]['inst_path'])

		if len(inst_mask.shape) == 3:
			# convert to sigle channel.
			inst_mask = rgb2sc(inst_mask)

		H, W = inst_mask.shape

		# init panoptic mask.
		pan_mask = np.zeros((H, W, 3))

		seg_info = []
		for i in np.unique(inst_mask): # for each object

			bm = to_binary_mask(inst_mask, i)
			
			indices = np.unravel_index(np.argmax(bm), bm.shape)
			catid = sem_mask[indices]
			if catid == 0: # unlabeled region.
				continue

			color = id_gen.get_color(catid)
			color_mask = np.ones((H, W, 3)) * color

			patch = cv2.bitwise_and(color_mask, color_mask, mask=bm)
			pan_mask = cv2.bitwise_or(pan_mask, patch)

			seg_info.append(
				{
					  'id': rgb2id(color)
					, 'category_id': catid
					, 'bbox': list(mask2bbox(bm))
					, 'area': mask2area(bm)
					, 'iscrowd': 0
				}
			)

		ann_name = names_and_paths[im_id]['ann_name']
		coco_d['annotations'].append(
			{
				"segments_info": seg_info,
				"file_name": ann_name,
				"image_id": im_id
			}
		)
	
		Image.fromarray(pan_mask.astype('uint8')).save(os.path.join(ann_dir, ann_name))

	return coco_d


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="This is a script that converts ACDC reference dataset to panoptic format.")
	parser.add_argument('-s', '--semantic_dir', type=str, help="the semantic annotations directory.", required=True)
	parser.add_argument('-i', '--instance_dir', type=str, help="the instance annotations directory.", required=True)
	parser.add_argument('-o', '--output_dir', type=str, help="the output directory.", required=True)
	parser.add_argument('--label_set', type=str, default='acdc', help="the label set to use from labels.py")
	parser.add_argument('-n', '--name', type=str, help="the name to be assigned to annotations directory.", default='coco-panoptic')
	parser.add_argument('--single_worker', action='store_true', help="Use only one process (for debug)", default=False)
	args = parser.parse_args()

	sem_dir = args.semantic_dir
	inst_dir = args.instance_dir
	out_dir = args.output_dir
	label_set = args.label_set

	json_file = os.path.join(out_dir, args.name + '.json')
	ann_dir =  os.path.join(out_dir, args.name)

	os.makedirs(ann_dir, exist_ok=True)

	main_worker(sem_dir, inst_dir, json_file, ann_dir, label_set, single_worker=args.single_worker)