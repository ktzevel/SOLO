import numpy as np
import trimesh
import os
from skimage.segmentation import find_boundaries

from calibration import gopro4, shutong


def depth2pcd(depth:np.ndarray, intrinsics:dict)->np.ndarray:
	""" 
		Args:
			depth (np.ndarray): the depth mask
			intrinsics (dict): a dict containing ['fx', 'fy', 'cx', 'cy']
		
		Returns:
			pcd (np.ndarray): a pointcloud of shape (N, 3)

			Based on:
			pc = d * inv(K) * augmented(ps)

			Orientation of the pointcloud should be:
			(-z)--(x)
			|
			(y)
	"""
	H, W = depth.shape

	fy = intrinsics['fy']
	fx = intrinsics['fx']
	cx = intrinsics['cx']
	cy = intrinsics['cy']

	xs = np.tile(range(W), H)
	ys = np.repeat(range(H), W)

	x = (xs - cx) / fx
	y = (ys - cy) / fy

	z = depth.flatten()

	return np.dstack((x * z, y * z, z)).reshape((-1, 3))


def save_pcd(path:str, pcd:np.ndarray):
	trimesh.PointCloud(pcd, process=False).export(path)


def save_mesh(path:str, m):
	m.export(path)
	
	# This will comment out the lines of the obj file
	# referring to materials.
	# Since trimesh does not have a way to include the UV
	# map without adding a material, this workaround is
	# used to achieve just that.
	if m.visual is not None:
		os.system(f'sed -i "s,mtllib,#mtllib," {path}')
		os.system(f'sed -i "s,usemtl,#usemtl," {path}')
	

# TODO: vectorize or save per resolution.
# TODO: add scale factor for more faces per pixel.
def pcd2mesh(pcd:np.ndarray, shape:tuple, rm_sky=True):
	def _idx(yy, xx):
		return yy * W + xx

	H, W = shape
	if not os.path.exists('lattice_grid.npy'):
		faces = [[_idx(ny, nx+1),   _idx(ny, nx),   _idx(ny+1, nx)]   for ny in range(0, H-1, 2) for nx in range(1, W-1, 2)] \
			+ [[_idx(ny, nx+1),   _idx(ny+1, nx), _idx(ny+1, nx+1)] for ny in range(0, H-1, 2) for nx in range(1, W-1, 2)] \
			+ [[_idx(ny, nx+1),   _idx(ny, nx),   _idx(ny+1, nx+1)] for ny in range(0, H-1, 2) for nx in range(0, W-1, 2)] \
			+ [[_idx(ny+1, nx+1), _idx(ny, nx),   _idx(ny+1, nx)]   for ny in range(0, H-1, 2) for nx in range(0, W-1, 2)] \
			+ [[_idx(ny, nx+1),   _idx(ny, nx),   _idx(ny+1, nx)]   for ny in range(1, H-1, 2) for nx in range(0, W-1, 2)] \
			+ [[_idx(ny, nx+1),   _idx(ny+1, nx), _idx(ny+1, nx+1)] for ny in range(1, H-1, 2) for nx in range(0, W-1, 2)] \
			+ [[_idx(ny, nx+1),   _idx(ny, nx),   _idx(ny+1, nx+1)] for ny in range(1, H-1, 2) for nx in range(1, W-1, 2)] \
			+ [[_idx(ny+1, nx+1), _idx(ny, nx),   _idx(ny+1, nx)]   for ny in range(1, H-1, 2) for nx in range(1, W-1, 2)]

		np.save('lattice_grid.npy', np.array(faces))
	else:
		faces = np.load('lattice_grid.npy')

	return trimesh.Trimesh(pcd, faces, process=False)

def remove_faces(mesh, depth, semantics, instances):
	verts = mesh.vertices
	faces = mesh.faces

	# ----------------------------------------------
	# faces containing vertices in the sky.
	# ----------------------------------------------
	#     sky: a flattened mask of a (1080, 1920) image indicating the pixels that correspond to the sky.
	#     sky[faces]: boolean values of shape (F, 3) indicating the verices of the faces that correspond to sky pixels.
	#     m: a mask of shape (F,1) indicating faces where any of their vertices correspond to sky pixels.
	#     we discard such faces.
	sky = ( semantics == 23 ).flatten()
	m = sky[faces].any(axis=-1)
	faces = faces[~m]

	return trimesh.Trimesh(verts, faces, process=False)


def project_uv(mesh, shape, intrinsics):
	""" Creates the UV map for the provided mesh.

		Args:
			mesh (trimesh.Trimesh): the mesh to process.
			shape (tuple): the shape of the image plane.
			intrinsics (dict): the camera calibration params.

		Returns:
			mesh (trimesh.Trimesh): the uv projected mesh.
		
		Description:
			The UV map provides a mapping between the vertices of the mesh
			and a 2d plane, called UV map, where u and v correspond to x and y axis respectively.
			To acquire such a map it is first required to project the 3d vertices on the 2d x-y plane.
			This is done by leveraging the calibration matrix parameters. The resulted x-y coordinates follow
			the ordering of the verices list of the mesh. This order can be arbitrary due to several additions
			and deletions that took place on the mesh. Therefore, the ordering of the vertices should be
			corrected such that the resulted x-y coordinates are ordered according to their values. Then the
			x-y coordinates should be normalized in [0,1] and the y axis should be flipped to arrive at the u-v
			coordinates that form the UV map.
	
			Example of how verices and faces are sorted:
				Vertices list:
				V : [  x3, x1, x2, ...]
				pos :  0   1   2

				idx = np.lexsort((xy[:,0], xy[:,1]))
				idx : [ 1, 2, 0]
				pos :   0  1  2
				0 -> 1
				1 -> 2
				2 -> 0

				inv_idx = np.argsort(idx)
				inv_idx : [ 2, 0, 1]
				pos :       0  1  2
				0 -> 2
				1 -> 0
				2 -> 1

				# xy = xy[idx]
				sort(V) : [  x1, x2, x3, ...]
				pos :        0   1   2

				Faces list:
				F : [ [1, _, 2], ..., [_, 0, 1], ... ]
				
				# map_idx = lambda i: inv_idx[i]
				# np.apply_along_axis(map_idx, 0, faces)
				updated(F):  [ [0, _, 1], ..., [_, 2, 0], ... ]

			XY and UV planes illustrated:
				XY plane:
					(0,0)---(1,0)
					|          |
					|          |
					(0,1)---(1,1)
				
				UV plane:
					(0,1)---(1,1)
					|          |
					|          |
					(0,0)---(1,0)
	"""
	H, W = shape
	vertices = mesh.vertices
	faces = mesh.faces

	fy = intrinsics['fy']
	fx = intrinsics['fx']
	cx = intrinsics['cx']
	cy = intrinsics['cy']

	# pin-hole model perspective projection.
	xc, yc, zc = vertices.transpose(1, 0)

	zero_mask = (zc <= 0.0)
	if zero_mask.any():
		zc[np.nonzero(zero_mask)] = 1e-6
		

	xs = ( fx * xc / zc ) + cx
	ys = ( fy * yc / zc ) + cy

	# clip points out of camera bounds.
	xs = np.clip(xs[...,None], a_min=0.0, a_max=W-1)
	ys = np.clip(ys[...,None], a_min=0.0, a_max=H-1)
	xy = np.hstack((xs, ys))

	# Implicit sort (using index array) using y as primary sorting key.
	idx = np.lexsort((xy[:,0], xy[:,1]))

	# Index for inverting the sorting.
	# Used to substitute vertices indexes in faces.
	inv_idx = np.argsort(idx)

	xy = xy[idx]

	# The order of the vertices should change accordingly.
	vertices = vertices[idx]
	
	# Updates the vertex indices in faces.
	map_idx = lambda i: inv_idx[i]
	faces = np.apply_along_axis(map_idx, 0, faces)

	# XY to UV plane.
	uv = np.zeros_like(xy)
	uv[:, 0] = xy[:, 0] / (W-1) # normalize x to get u.
	uv[:, 1] = ((H-1) - xy[:, 1]) / (H-1) # change the direction of incr. of y and normalize to get v.

	# adds UV coordinates to the mesh.
	vis = trimesh.visual.texture.TextureVisuals(uv=uv)
	mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=vis)
	return mesh


def get_mesh(depth:np.ndarray, semantics:np.ndarray, intrinsics, rm_sky=True, store_pcd=False):

	depth = depth.copy()

	# Create a pointcloud using depth for back-projection.
	pcd = depth2pcd(depth, intrinsics)

	if store_pcd:
		# substitutes -1 in z coords with inf.
		epcd = pcd.copy()
		m = (epcd[...,-1] == -1)
		epcd[m] = np.inf
		save_pcd('pointcloud.obj', epcd)

	mesh = pcd2mesh(pcd, depth.shape, rm_sky)
	return mesh
