"""
	FOR BLENDER v3.4
"""

import bpy
from bpy import context, data, ops
import os
import argparse
import sys
import json
import copy

import mathutils
import math
from math import radians
from mathutils import Euler

import bmesh

import random
import re

# Those are kept as comment for debugging with Blender's user interface.
# mesh_path = "/home/ktzevel/Documents/ETH_Project/Code/night_simulation/ray-tracing/data/mesh/final.obj"
# albedo_path = "/home/ktzevel/Documents/ETH_Project/Code/night_simulation/ray-tracing/data/albedo/albedo.png"
# roughnesh_path = "/home/ktzevel/Documents/ETH_Project/Code/night_simulation/ray-tracing/data/roughness/roughness.png"
# lights_dir = "/home/ktzevel/Documents/ETH_Project/Code/night_simulation/ray-tracing/data/lights/GOPR0351_frame_000159"
# out_dir = "/home/ktzevel/Documents/ETH_Project/Code/night_simulation/ray-tracing/output/rendered.png"


def reset_scene():
	c = context.copy()
	c["selected_objects"] = list(context.scene.objects)
	with context.temp_override(**c):
		bpy.ops.object.delete()

	# set world's background to black.
	wn = context.scene.world.name
	data.worlds[wn].node_tree.nodes["Background"].inputs[0].default_value=(0,0,0,1)


def set_renderer(out_path, max_bounces, samples, time_limit, noise, indirect_clamp, use_denoising, xyz):

	scene = data.scenes[0]
	scene.render.engine = 'CYCLES'

	context.preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
	scene.cycles.device = 'GPU'

	context.preferences.addons["cycles"].preferences.get_devices()
	print(context.preferences.addons["cycles"].preferences.compute_device_type)

	for d in context.preferences.addons["cycles"].preferences.devices:
		if not "AMD" in d['name']:
			d["use"] = 1
	
	for d in context.preferences.addons["cycles"].preferences.devices:
		print(d["name"], d["use"])

	scene.frame_end = 1
	scene.render.resolution_x = 1920
	scene.render.resolution_y = 1080
	scene.render.resolution_percentage = 100
	scene.render.image_settings.file_format = 'PNG'
	scene.render.filepath = out_path
	
	scene.cycles.max_bounces = max_bounces
	scene.cycles.samples = samples
	scene.cycles.time_limit = time_limit
	scene.cycles.adaptive_threshold = noise
	scene.cycles.use_denoising = use_denoising
	scene.cycles.sample_clamp_indirect = indirect_clamp

	if xyz:
		scene.render.image_settings.file_format = 'TIFF'
		scene.display_settings.display_device = 'XYZ'
		scene.view_settings.view_transform = 'Standard'
		scene.view_settings.look = 'None'
		scene.render.image_settings.color_mode = 'RGB'
		scene.render.image_settings.color_depth = '16'
		scene.render.image_settings.tiff_codec = 'NONE'


def set_camera(exposure:float, gamma:float):
	ops.object.camera_add(rotation=(0, radians(180), radians(180)))
	cam = bpy.data.cameras[-1]
	cam.lens = 33.5
	cam.sensor_width = 36.0

	scene = data.scenes[0]
	# exposure
	scene.view_settings.exposure = exposure
	# gamma
	scene.view_settings.gamma = gamma


def load_mesh(path):
	ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")


def select_object(handle):
	for o in context.scene.objects:
		o.select_set(False)

	handle.select_set(True)
	context.view_layer.objects.active = handle


def get_area(t:str):
	screen = bpy.context.workspace.screens.get('Scripting')
	for a in screen.areas:
		if a.type == t:
			return a


def get_area_region(area, t:str):
	for r in area.regions:
		if r.type == t:
			return r


def enable_camera_view():
	context.scene.camera = context.scene.objects.get('Camera')
	get_area('VIEW_3D').spaces.active.region_3d.view_perspective = 'CAMERA'


def create_uv(mesh):
	# BUG: This needs a graphical interface to work.
	a = get_area('VIEW_3D')
	r = get_area_region(a, 'WINDOW')
	w = context.window
	with context.temp_override(window=w, area=a, region=r):
		enable_camera_view()
		select_object(mesh)
		bpy.ops.object.mode_set(mode='EDIT')
		bpy.ops.mesh.select_all(action = 'SELECT')
		bpy.ops.uv.project_from_view(clip_to_bounds=True, scale_to_bounds=True)
		bpy.ops.object.mode_set(mode='OBJECT')

def visit_faces(mesh_name:str):
	""" method for iterating through all the faces and accessing parameters such as uv coords. """
	me = data.meshes[mesh_name]
	bm = bmesh.new()
	bm.from_mesh(me)

	uv_lay = bm.loops.layers.uv.active

	bm.faces.ensure_lookup_table()
	for face in bm.faces:
		for loop in face.loops:
			uv = loop[uv_lay].uv
			print("Loop UV: %f, %f" % uv[:])
			vert = loop.vert
			print("Loop Vert: (%f,%f,%f)" % vert.co[:])

	bm.to_mesh(me)
	bm.free()

def set_bsdf_defaults(bsdf):
	bsdf.distribution = "MULTI_GGX"
	bsdf.subsurface_method = "RANDOM_WALK"

	bsdf.inputs['IOR'].default_value = 0.0 # value is ignored since Transmission is set to zero.
	bsdf.inputs['Specular'].default_value = 0.0 # value concerns dielectrics.
	bsdf.inputs['Subsurface IOR'].default_value = 0.0 # For non-metals the most common is 0.5 (Blender's default)
	bsdf.inputs['Sheen Tint'].default_value = 0.0
	bsdf.inputs['Clearcoat Roughness'].default_value = 0.0
	bsdf.inputs['Subsurface Radius'].default_value[0] = 0.0 # we have no information for subsurface reflection
	bsdf.inputs['Subsurface Radius'].default_value[1] = 0.0 # we set those to zero.
	bsdf.inputs['Subsurface Radius'].default_value[2] = 0.0
	# other inputs are set to 0.0 by default.


def multi_add(ops):
	""" 
		Adds at least two operators.
		Extends the functionality of the ADD node which supports only two operators.
	"""
	def new_ADD():
		add = nodes.new("ShaderNodeVectorMath")
		add.operation = 'ADD'
		return add

	if len(ops) < 1:
		raise ValueError('multi_add for strength nodes called without inputs.')

	if len(ops) == 1:
		val = nodes.new(type="ShaderNodeValue")
		val.outputs[0].default_value = 0.0
		add = new_ADD()
		node_tree.links.new(add.inputs[0], ops[0].outputs[0])
		node_tree.links.new(add.inputs[1], val.outputs['Value'])
		return add

	add_prev = None
	add = new_ADD()
	node_tree.links.new(add.inputs[0], ops[0].outputs[0])
	node_tree.links.new(add.inputs[1], ops[1].outputs[0])

	if len(ops) == 2:
		return add
	
	for i in range(2, len(ops)):
		add_prev = add
		add = new_ADD()
		node_tree.links.new(add_prev.outputs[0], add.inputs[0])
		node_tree.links.new(ops[i].outputs[0], add.inputs[1])

	return add

def squared_cos(input_node):
	cos1 = nodes.new("ShaderNodeVectorMath")
	cos1.operation = 'COSINE'

	cos2 = nodes.new("ShaderNodeVectorMath")
	cos2.operation = 'COSINE'

	mul = nodes.new("ShaderNodeVectorMath")
	mul.operation = 'MULTIPLY'

	node_tree.links.new(cos1.inputs[0], input_node.outputs['Result'])
	node_tree.links.new(cos2.inputs[0], input_node.outputs['Result'])
	node_tree.links.new(cos1.outputs[0], mul.inputs[0])
	node_tree.links.new(cos2.outputs[0], mul.inputs[1])

	return mul


def strength_node(path:str, spread:str):

	s_tex = nodes.new("ShaderNodeTexImage")
	s_tex.image = bpy.data.images.load(path)
	s_tex.image.colorspace_settings.name = 'Non-Color'

	if spread is None:
		val = nodes.new(type="ShaderNodeValue")
		val.outputs[0].default_value = 1.0

		mult = nodes.new("ShaderNodeVectorMath")
		mult.operation = 'MULTIPLY'

		node_tree.links.new(mult.inputs[0], val.outputs['Value'])
		node_tree.links.new(mult.inputs[1], s_tex.outputs['Color'])

	elif spread == 'cos':
		geom = nodes.new("ShaderNodeNewGeometry")
		dot = nodes.new("ShaderNodeVectorMath")
		dot.operation = 'DOT_PRODUCT'

		node_tree.links.new(geom.outputs['Normal'], dot.inputs[0])
		node_tree.links.new(geom.outputs['Incoming'], dot.inputs[1])
	
		rmap = nodes.new(type="ShaderNodeMapRange")
		rmap.inputs['To Min'].default_value = -math.pi/2
		rmap.inputs['To Max'].default_value = 0.0

		node_tree.links.new(rmap.inputs['Value'], dot.outputs['Value'])

		cos = nodes.new("ShaderNodeVectorMath")
		cos.operation = 'COSINE'
		node_tree.links.new(rmap.outputs['Result'], cos.inputs[0])

		mult = nodes.new("ShaderNodeVectorMath")
		mult.operation = 'MULTIPLY'

		node_tree.links.new(mult.inputs[0], cos.outputs[0])
		node_tree.links.new(mult.inputs[1], s_tex.outputs['Color'])

	elif spread == 'cos2':
		geom = nodes.new("ShaderNodeNewGeometry")
		dot = nodes.new("ShaderNodeVectorMath")
		dot.operation = 'DOT_PRODUCT'

		node_tree.links.new(geom.outputs['Normal'], dot.inputs[0])
		node_tree.links.new(geom.outputs['Incoming'], dot.inputs[1])
	
		rmap = nodes.new(type="ShaderNodeMapRange")
		rmap.inputs['To Min'].default_value = -math.pi/2
		rmap.inputs['To Max'].default_value = 0.0

		node_tree.links.new(rmap.inputs['Value'], dot.outputs['Value'])
		scos = squared_cos(rmap)

		mult = nodes.new("ShaderNodeVectorMath")
		mult.operation = 'MULTIPLY'

		node_tree.links.new(mult.inputs[0], scos.outputs[0])
		node_tree.links.new(mult.inputs[1], s_tex.outputs['Color'])
	else:
		raise ValueError('This type of spread is not supported.')

	return mult

def set_ego_vehicle_lights():
	light_data = bpy.data.lights.new(name="light-props", type='AREA')
	light_data.energy = 100 # in Watts
	light_data.size = 0.3 # x-axis
	light_data.size_y = 0.15 # y-axis
	light_data.color = (0.516888, 0.491321, 0.330107) # RGB sampled from night-time illuminants.

	# Spread should be greater or equal to 125 deg.
	bpy.data.lights['light-props'].spread = radians(125) # in degrees
	bpy.data.lights['light-props'].shape = "RECTANGLE" # shape of the headlight

	# create light objects:
	headlight_l = bpy.data.objects.new(name="head_L", object_data=light_data)
	headlight_r = bpy.data.objects.new(name="head_R", object_data=light_data)

	# Link object to collection in context
	bpy.context.collection.objects.link(headlight_l)
	bpy.context.collection.objects.link(headlight_r)

	# Change lights position
	# x: horizontal axis (0.5m away from the middle of the vehicle)
	# y: verical axis (0.3m below camera)
	# z: goes into image plane (more than 0.5m from the camera)
	headlight_r.location = (0.5, 0.3, 0.5)
	
	# X must be more than 133 deg. above the direction of the vehicle.
	# Y must be more than (+/-) 17.5 deg. away from the direction of the vehicle.
	headlight_r.rotation_euler = Euler((radians(135), radians(18), radians(0)), 'XYZ')

	headlight_l.location = (-0.5, 0.3, 0.5)
	headlight_l.rotation_euler = Euler((radians(135), radians(-18), radians(0)), 'XYZ')

def world_sun_light_setup(sun_config=None, strength=1.0):
	world_node_tree = bpy.context.scene.world.node_tree
	world_node_tree.nodes.clear()

	node_sky = world_node_tree.nodes.new(type="ShaderNodeTexSky")
	world_background_node = world_node_tree.nodes.new(type="ShaderNodeBackground")
	world_background_node.inputs["Strength"].default_value = strength
	world_output_node = world_node_tree.nodes.new(type="ShaderNodeOutputWorld")

	if sun_config:
		for attr, value in sun_config.items():
			if hasattr(node_sky, attr):
				setattr(node_sky, attr, value)
			else:
				print("\t warning: %s is not an attribute of ShaderNodeTexSky node", attr)

		world_node_tree.links.new(node_sky.outputs["Color"], world_background_node.inputs["Color"])
		world_node_tree.links.new(world_background_node.outputs["Background"], world_output_node.inputs["Surface"])


def compositor_setup():
	""" 
		For imitating glare effect.
		Blender's Compositor gets executed after the rendering stage.
	"""

	scene = bpy.context.scene
	scene.use_nodes = True
	compositor_node_tree = scene.node_tree
	nodes = compositor_node_tree.nodes

	# image_node = nodes.new(type="CompositorNodeImage")
	image_node = nodes.new(type="CompositorNodeRLayers")
	glare = nodes.new("CompositorNodeGlare")
	glare.glare_type = 'FOG_GLOW'
	glare.threshold = 0.0
	glare.size = 9
	glare.quality = "HIGH"

	viewer_node = compositor_node_tree.nodes.new(type="CompositorNodeViewer")
	composite_node = compositor_node_tree.nodes.new(type="CompositorNodeComposite")

	compositor_node_tree.links.new(image_node.outputs["Image"], glare.inputs["Image"])
	compositor_node_tree.links.new(glare.outputs["Image"], composite_node.inputs["Image"])
	compositor_node_tree.links.new(glare.outputs["Image"], viewer_node.inputs["Image"])

def set_street_lights(stl_l:list, extra=False):
	''' Adds the street lights using the params provided in the street light list. 
		stl_l (list): a list of street lights and their properties.
		extra (bool): if true an extra light is created over the camera at (X=0.0 Y=5.0 Z=[-2.0, 12.0])
	'''

	def create_emission_material(rgba, strength):

		# if this name already exists Blender will create a new one (e.g. mtl_emission.001)
		mat = bpy.data.materials.new(name='mtl_emission')

		mat.use_nodes=True
		nodes = mat.node_tree.nodes
		links = mat.node_tree.links

		# clear default nodes:
		for n in nodes:
			nodes.remove(n)
		
		nodes.new('ShaderNodeOutputMaterial')
		nodes.new('ShaderNodeEmission')
		geom = nodes.new('ShaderNodeNewGeometry')
		dot = nodes.new('ShaderNodeVectorMath')
		dot.operation = 'DOT_PRODUCT'

		links.new(geom.outputs['Normal'], dot.inputs[0])
		links.new(geom.outputs['Incoming'], dot.inputs[1])
	
		# dot product range is [0,1]
		# maps from [0,1]->[-pi/2, 0]
		# since cos is an increasing function in that area.
		# dot = 0 means that view and normal vectors are perpendicular.
		rmap = nodes.new(type='ShaderNodeMapRange')
		rmap.inputs['To Min'].default_value = -math.pi/2
		rmap.inputs['To Max'].default_value = 0.0

		links.new(rmap.inputs['Value'], dot.outputs['Value'])

		cos = nodes.new('ShaderNodeVectorMath')
		cos.operation = 'COSINE'
		links.new(rmap.outputs['Result'], cos.inputs[0])

		mult = nodes.new('ShaderNodeMath')
		mult.operation = 'MULTIPLY'

		links.new(mult.inputs[0], cos.outputs[0])
		mult.inputs[1].default_value = strength

		# rgb should already be converted in sRGB.
		nodes['Emission'].inputs[0].default_value = rgba
		links.new(mult.outputs['Value'], nodes['Emission'].inputs[1])

		# links emission node to material's surface.
		links.new(nodes['Emission'].outputs[0], nodes['Material Output'].inputs['Surface'])

		return mat

	def create_diffuse_material(rgba):

		# if this name already exists Blender will create a new one (e.g. mtl_emission.001)
		mat = bpy.data.materials.new(name='mtl_diffuse')

		mat.use_nodes=True
		nodes = mat.node_tree.nodes
		links = mat.node_tree.links

		# clear default nodes:
		for n in nodes:
			nodes.remove(n)
		
		# output material node.
		nodes.new('ShaderNodeOutputMaterial')
		nodes.new('ShaderNodeBsdfDiffuse')

		# rgb should already be converted in sRGB.
		nodes['Diffuse BSDF'].inputs[0].default_value = rgba

		# links Diffuse BSDF node to material's surface.
		links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs['Surface'])

		return mat

	#------------------------------------------

	# Adds an extra street_light.
	if extra:

		# minimum distance from the next street light.
		stl_inter_dist = 20

		# find the ligth closer to the camera.
		nearest_z = 500
		for s in stl_l:
			if s['coords'][2] < nearest_z:
				nearest_z = s['coords'][2]
		
		max_z = 13.0 # light should not be visible.
		min_z = -2

		ref_max_z = nearest_z - stl_inter_dist
		if ref_max_z > min_z:
			
			if not stl_l:
				extra_stl = {
					"coords": [],
					"strength": 950.0,
					"rgb": [ 0.1570216384117171, 0.0830302131269754, 0.009400825796746838 ],
					"size": 0.7014957904815674
				}
			else:
				extra_stl = copy.deepcopy(stl_l[0])

			x = 0.0   # m
			y = -5.0  # m
			z = random.uniform(min_z, min(ref_max_z, max_z))  # m
			extra_stl['coords'] = [x, y, z]
			stl_l.append(extra_stl)

	for i, entry in enumerate(stl_l):
		coords = entry['coords']
		strength = entry['strength']
		rgb = entry['rgb']
		size = entry['size']
	
		# create light's mesh.
		bpy.ops.mesh.primitive_cube_add(
			  size=size
			, location=coords
			, scale=(1.0, 0.01, 1.0) # scales down y-axis.
		)
		
		# select newly created obj and rename it accordingly.
		cube = bpy.context.active_object
		cube.name = f'cube_{i}'
		
		# find cube's lower face along y-axis.
		# y-axis 0 is located at the top of the image.
		y_max = 0.0 # init.
		bidx = None
		for i, face in enumerate(cube.data.polygons):
			y = face.center[1] # gets face center's Y coord. (Y=1)
			if y > y_max:
				y_max = y
				bidx = i
		
		# add material slot
		bpy.ops.object.material_slot_add()

		# set material slot.
		rgba = rgb + [1] # [r g b] + [1] = [r g b a=1]
		cube.material_slots[0].material = create_emission_material(rgba, strength)

		bottom_face = cube.data.polygons[bidx]
		bottom_face.select = True

		bpy.ops.object.mode_set(mode='EDIT')

		bpy.ops.object.material_slot_assign()
		bpy.ops.mesh.select_all(action='DESELECT') 

		bpy.ops.object.mode_set(mode='OBJECT')

		# set rest of the faces.
		bpy.ops.object.material_slot_add()

		# create a diffuse black material for street lights exterior.
		# or reuse existing one.
		rgba = [0, 0, 0, 1]
		mat = create_diffuse_material(rgba)

		# set diffuse material in the second mat. slot of the mesh light.
		cube.material_slots[1].material = mat

		# apply the second material slot to all the faces except of the bottom one.
		for i, face in enumerate(cube.data.polygons):
			if i == bidx:
				continue # skip bottom face.

			face.select = True
			bpy.ops.object.mode_set(mode='EDIT')
			bpy.ops.object.material_slot_assign()
			bpy.ops.mesh.select_all(action='DESELECT') 
			bpy.ops.object.mode_set(mode='OBJECT')


class BlenderArgParser(argparse.ArgumentParser):
	""" 
		Shortcut for passing command line arguments to a python script excecuted by Blender.
		Adapted from:
		https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
	"""

	def _get_argv_after_doubledash(self):
		if "--" in sys.argv:
			idx = sys.argv.index("--")

			# returns the args after '--'
			return sys.argv[idx+1:]
		else:
			return sys.argv

	def parse_args(self):
		return super().parse_args(args=self._get_argv_after_doubledash())

"""
	Directionality dictionary:
	For each category a directionality type is assigned.
"""
cat2spread = {
	  'unlabeled': None
	, 'window_building': None
	, 'window_parked': None
	, 'window_transport': None
	, 'traffic_light_G': None
	, 'traffic_light_O': None
	, 'traffic_light_R': None
	, 'street_light_HT': None # we use artificial ones
	, 'street_light_LT': None # check set_street_lights()
	, 'parked_front': 'cos'
	, 'parked_rear': None
	, 'moving_front': 'cos'
	, 'moving_rear': None
	, 'advertisement': None
	, 'clock': None
	, 'inferred': None
}


if __name__ == "__main__":

	parser = BlenderArgParser(description="Blender ray-tracing script.")
	parser.add_argument('mesh', type=str, help="mesh")
	parser.add_argument('albedo', type=str, help="albedo image texture")
	parser.add_argument('roughness', type=str, help="roughness image texture")
	parser.add_argument('lights_dir', type=str, help="lights per image directory.")
	parser.add_argument('output_dir', type=str, help="output directory")

	parser.add_argument('--save_blend', action="store_true", help="stores .blend file.")
	parser.add_argument('--use_denoising', action="store_true", default=False, help="denoises the rendered image.")
	parser.add_argument('--xyz', action="store_true", default=False, help="returns a linear XYZ image.")


	parser.add_argument('--max_bounces', default=12, type=int, help="sets max rendering samples.")
	parser.add_argument('--samples', default=4096, type=int, help="sets max rendering samples.")
	parser.add_argument('--time_limit', default=3600, type=int, help="sets max rendering time in seconds.")
	parser.add_argument('--noise', default=0.01, type=float, help="noise thresshold.")
	parser.add_argument('--indirect_clamp', default=10, type=float, help="clamps indirect sample values.")
	parser.add_argument('--exposure', default=0.0, type=float, help="sets exposure in camera settings.")
	parser.add_argument('--gamma', default=1.0, type=float, help="sets gamma in camera settings.")
	parser.add_argument('--seed', default=1432, type=int, help="sets random seed.")

	parser.add_argument('--twilight_sky', action="store_true", help="Whether to render a twilight sky")


	args = parser.parse_args()

	mesh_path = args.mesh
	albedo_path = args.albedo
	roughnesh_path = args.roughness
	out_dir = args.output_dir
	lights_dir = args.lights_dir

	exposure = args.exposure
	gamma = args.gamma
	max_bounces = args.max_bounces
	samples = args.samples
	time_limit = args.time_limit

	twilight_sky = args.twilight_sky


	# Random seed for z-component of the extra street light.
	# The code re-runs for each image.
	# seed arg sets the base seed upon which we add the num part of the corresponding fid.
	fid_num_part = re.sub('[^0-9]', '', os.path.basename(mesh_path))
	seed = args.seed
	random.seed(seed + int(fid_num_part))

	noise = args.noise
	use_denoising = args.use_denoising

	indirect_clamp = args.indirect_clamp

	save_blend = args.save_blend
	xyz = args.xyz
	if xyz:
		gamma = 1.0
		exposure = 0.0

	os.makedirs(out_dir, exist_ok=True)

	
	reset_scene() # clears every object in the scene.
	load_mesh(mesh_path)

	fid = os.path.basename(albedo_path).split('.')[0]
	out_path = os.path.join(out_dir, f'{fid}.png')
	set_renderer(out_path, max_bounces, samples, time_limit, noise, indirect_clamp, use_denoising, xyz)

	set_camera(exposure, gamma)

	mesh = context.scene.objects[0]
	mat = bpy.data.materials.new(name="material")
	mat.use_nodes = True
	mesh.data.materials.append(mat)

	node_tree = mat.node_tree
	nodes = node_tree.nodes

	bsdf = nodes.get("Principled BSDF")
	set_bsdf_defaults(bsdf)

	albedo_tex = nodes.new("ShaderNodeTexImage")
	albedo_tex.image = bpy.data.images.load(albedo_path)
	albedo_tex.image.colorspace_settings.name = 'sRGB'

	roughness_tex = nodes.new("ShaderNodeTexImage")
	roughness_tex.image = bpy.data.images.load(roughnesh_path)
	roughness_tex.image.colorspace_settings.name = 'Non-Color'

	xyz_tex = nodes.new("ShaderNodeTexImage")
	xyz_tex.image = bpy.data.images.load(os.path.join(lights_dir, 'xyz.png'))
	xyz_tex.image.colorspace_settings.name = 'XYZ'
	
	"""
		Light strength:
		- That includes a base strength and directionality.
		- The strength and directionality is set for each category.
		- A B&W mask is given per category as a base strength (different strength may apply to each instance.)
		- A str denoting directionality type per category.
	"""
	strength_dir = os.path.join(lights_dir, 'strength')

	fn_l = os.listdir(strength_dir)
	cat_l = [fn.split('.')[0] for fn in fn_l]
	fncat_l = list(zip(fn_l, cat_l))

	strength_node_l = []
	for fn, cat in fncat_l:
		path = os.path.join(strength_dir, fn)
		sn = strength_node(path, cat2spread[cat])
		strength_node_l.append(sn)

	if not strength_node_l:
		# Creates a place-holder in case there is no strength defined.
		# That can happen when the image has only street lights.
		emission_strength = nodes.new("ShaderNodeValue")
	else:
		emission_strength = multi_add(strength_node_l)

	node_tree.links.new(albedo_tex.outputs['Color'], bsdf.inputs['Base Color'])
	node_tree.links.new(roughness_tex.outputs['Color'], bsdf.inputs['Roughness'])
	node_tree.links.new(xyz_tex.outputs['Color'], bsdf.inputs['Emission'])
	node_tree.links.new(emission_strength.outputs[0], bsdf.inputs['Emission Strength'])

	# street lights are handled separately.
	stl_json = os.path.join(lights_dir, 'street_lights.json')
	stl_l = []
	with open(stl_json) as fp:
		stl_l = json.load(fp)

	# creates a street light for each entry of the json file.
	set_street_lights(stl_l, extra=True)

	set_ego_vehicle_lights()

	# render twilight sky.
	if twilight_sky:
		# The Civil Twilight corresponds to a range from -6 to 0 degrees.
		# altitude is set to 408m which is the average in the Zurich City where ACDC was captured.
		elevation = random.uniform(-5.5, -3.5)
		azimuth = random.uniform(0, 360)
		cnf = { 
				  'sun_elevation': radians(elevation)
				, 'sun_rotation': radians(azimuth)
				, 'altitude': 408,

		}
		world_sun_light_setup(sun_config=cnf, strength=1.0)

	if save_blend:
		path = os.path.join(out_dir, f'{fid}.blend')
		bpy.ops.wm.save_as_mainfile(filepath=path)

	# for imitating fog glare.
	compositor_setup()

	# Start rendering
	context.scene.camera = context.scene.objects.get('Camera')
	ops.render.render(write_still=True)
