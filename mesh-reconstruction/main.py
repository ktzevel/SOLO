import os
import logging
import trimesh
import PIL.Image as Image
import numpy as np

from uncertain import uncertain, modify_pole
from depth_refinement import DepthRefine
from post_process_avg import PostProcessAVG
from handle_intersection import Handle_intersection

from get_mesh import get_mesh, save_mesh, project_uv, remove_faces
from options import Options
import calibration
from skimage.filters import gaussian, sobel, butterworth

import matplotlib.pyplot as plt

import multiprocessing
from tqdm import tqdm

YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format=YELLOW + "%(asctime)s - %(levelname)s - " + GREEN + "%(message)s" + RESET, datefmt="%Y-%m-%d %H:%M:%S")

class GeometricReconstruct:
	def __init__(self, args):
		self.opt = args
		self.convert_to_meters = self.opt['convert_to_meters']

		self.sem_dir = self.opt['semantics_dir']
		self.inst_dir = self.opt['instances_dir']
		self.depth_dir = self.opt['depth_dir']
		self.normal_dir = self.opt['normal_dir']
		self.rgb_dir = self.opt['rgb_dir']

		self.intrinsics = calibration.__dict__[self.opt['calibration']]

		self.out_dir = self.opt['out_dir']
		self.mesh_final_dir = os.path.join(self.out_dir, 'meshes', 'final')

		depth_fn = os.path.basename(self.depth_dir)
		self.refined_depth_dir = os.path.join(self.out_dir, depth_fn + '-refined')

		os.makedirs(self.mesh_final_dir, exist_ok=True)
		os.makedirs(self.refined_depth_dir, exist_ok=True)

	def single_image(self, fid):

		logging.info(f"Image: {fid}")
		
		rgb_path = os.path.join(self.rgb_dir, fid + '.png')
		rgb = np.array(Image.open(rgb_path))

		depth_path = os.path.join(self.depth_dir, fid + '.npy')
		depth = np.load(depth_path)
		
		# used for DPT dataset.
		if self.convert_to_meters:
			depth = depth * 1e-3

		normal_path = os.path.join(self.normal_dir, fid + '.npy')
		normals = np.load(normal_path).transpose([1, 2, 0])

		inst_path = os.path.join(self.inst_dir, fid + '.png')
		instances = np.array(Image.open(inst_path))

		sem_path = os.path.join(self.sem_dir, fid + '.png')
		semantics = np.array(Image.open(sem_path))
		semantics[semantics == 0] = 100 # null signs?

		logging.info("Generation of the uncertainty map.")
		uncertain_map, uncertain_map_opt = uncertain( self.opt, semantics, depth)

		logging.info("Depth refinement using normals.")
		refine_depth = DepthRefine( self.opt, depth, normals, uncertain_map_opt, semantics)
		depth = refine_depth.optimize()

		logging.info("Handle depth in intersections.")
		handle_intersection = Handle_intersection( depth, uncertain_map, semantics)
		depth = handle_intersection.handle_intersection()

		logging.info("Modify Pole.")
		depth = modify_pole(self.opt, semantics, depth)

		# saves refined depth.
		np.save(os.path.join(self.refined_depth_dir, f'{fid}.npy'), depth)

		logging.info("Generate mesh based on depth.")
		mesh = get_mesh(depth, semantics, self.intrinsics, rm_sky=True)
		
		# removes sky faces.
		mesh = remove_faces(mesh, depth, semantics, instances)

		logging.info("Mesh post-processing.")
		post_process = PostProcessAVG(self.opt, mesh, uncertain_map, semantics)
		post_process.remove_unexpected()
		vertices, faces = post_process.mesh_compeletion()
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

		logging.info("Project UV")
		mesh = project_uv(mesh, (1080, 1920), self.intrinsics)
		save_mesh(os.path.join(self.mesh_final_dir, f'{fid}.obj'), mesh)

	def all(self):
		get_fid = lambda p: ('_'.join(p.split('_')[:3]).split('.')[0])
		fids = [get_fid(img) for img in os.listdir(self.depth_dir)]
		for fid in fids:
			self.single_image(fid)

if __name__ == '__main__':

	args = Options().parse()
	fid = args['image_id']
	all_images = args['all']

	g = GeometricReconstruct(args)
	if all_images:
		logging.warning("Running in batch mode. All available images will be processed.")
		g.all()
	else:
		logging.warning("Running in single image mode.")
		g.single_image(fid)
