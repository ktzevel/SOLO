from typing import List, Any
import numpy as np
import rawpy
from exiftool import ExifToolHelper # pip install PyExifTool # https://sylikc.github.io/pyexiftool/examples.html
import matplotlib.pyplot as plt

from cct_utils import interpolate_cst, xyz2cct
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007


def get_metadata(img_path:str) -> dict:

	def list2arr(cm):
		return np.reshape(np.asarray(cm, dtype='float64'), (3, 3))
	
	def int_list(val):
		return [int(v) for v in val]

	def float_list(val):
		return [float(v) for v in val]

	def str2list(val):
		return val.split(' ')

	tag_map = {
		  'EXIF:ActiveArea': 'active_area'
		, 'EXIF:DefaultCropSize': 'default_size'
		, 'MakerNotes:PerChannelBlackLevel': 'black_level'
		, 'EXIF:WhiteLevel': 'white_level'
		, 'EXIF:CFAPattern2': 'cfa_pattern'
		, 'EXIF:ISO': 'iso'
		, 'EXIF:ExposureTime': 'exposure_time'
		, 'EXIF:ApertureValue': 'aperture'
		, 'EXIF:FocalLength': 'focal_length'
		, 'EXIF:ColorMatrix1': 'cm1'
		, 'EXIF:ColorMatrix2': 'cm2'
	}

	with ExifToolHelper() as et:
		exif_d = et.get_tags(img_path, tags=list(tag_map.keys()))[0]
	
	metadata = {}
	metadata = {tag_map[k]:v for k, v in exif_d.items() if k in list(tag_map.keys())}

	metadata['active_area'] = int_list(str2list(metadata['active_area']))
	metadata['default_size'] = int_list(str2list(metadata['default_size']))[::-1]
	metadata['black_level'] = int_list(str2list(metadata['black_level']))
	metadata['cfa_pattern'] = int_list(str2list(metadata['cfa_pattern']))
	metadata['aperture'] = round(metadata['aperture'],2)

	cm1 = float_list(str2list(metadata['cm1']))
	cm2 = float_list(str2list(metadata['cm2']))

	metadata['cm1'] = list2arr(cm1)
	metadata['cm2'] = list2arr(cm2)

	return metadata


def raw_cropping(img:np.ndarray, active_area:List[int], default_size:List[int]) -> np.ndarray:
	"""
		Crops pixels exposed to light.

		Args:
			img (np.ndarray): the image to crop.
			active_area (list): active area specification [ top, left, bottom, right ]
			default_size (list): default area specification [ bottom, right ]

			example:
				H, W = img.shape
				active_area =  [ 0, 0, H, W ]
		
		Returns:
			the cropped image.
	"""
	y_origin = active_area[0]
	x_origin = active_area[1]

	height = default_size[0]
	width = default_size[1]
	return img[y_origin:y_origin + height, x_origin:x_origin + width]


def normalize(img:np.ndarray, black_level, white_level) -> np.ndarray:
	""" 
		Black light subtraction plus normalization.

		Args:
			img (np.ndarray): the image to normalize, this image must be in bayer form.
			black_level: specifies the smallest sensor reading value.
			white_level: specifies the highest sensor reading value.
		
		Returns:
			the clipped normalized image.
	"""
	black_level_mask = np.zeros(img.shape)

	idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
	step2 = 2
	for i, idx in enumerate(idx2by2):
		black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]

	img = img.astype(np.float64)
	img = ( img - black_level_mask ) / ( white_level - black_level_mask )
	img = np.clip(img, a_min=0.0, a_max=1.0)
	return img


def demosaic(img:np.ndarray, cfa_pattern:List[int]) -> np.ndarray:
	"""
		Demosaicing Based Onwavelet Analysis of the Luminance Component
		https://ieeexplore.ieee.org/abstract/document/4379122

		Args:
			img (np.ndarray): the image for demosaicing, this image must be in bayer form.
			cfa_pattern (list): specifies the order of the channels.

		Returns:
			The demosaiced RGB image.
	"""
	max_val = 65535
	img = (img * max_val).astype(dtype=np.uint16)

	cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])

	img = demosaicing_CFA_Bayer_Menon2007(img, pattern=cfa_pattern_str)
	img = img.astype(dtype=np.float64) / max_val

	return img


def find_opt_cm(wp:List[float], cm1:np.ndarray, cm2:np.ndarray) -> np.ndarray:
	"""
		Args:
			cm1 (np.ndarray): adobe DNG color matrix for standard illuminant A.
			cm2 (np.ndarray): adobe DNG color matrix for standard illuminant D65.
			wp (list): the wp to calculate the interpolated color matrix for, expressed in CIE's XYZ colorspace.
		
		Results:
			A typically normalized color matrix capable of transforming from XYZ_wp to RGB_wp.
	"""

	def wp_cm_norm(cm, wp):
		"""
			Typical normalization for color matrices.
			Color matrices are typically normalized so that the characterization illuminant (wp),
			expressed in CIE's XYZ color space, just saturates in the camera space.
		"""
		maxval = np.max(cm @ wp)
		return (1.0 / maxval) * cm

	wp = np.reshape(np.asarray(wp), (3, 1)) # wp should be a column vec.
	cm1 = wp_cm_norm(cm1, wp)
	cm2 = wp_cm_norm(cm2, wp)
	cm = interpolate_cst(cm1, cm2, xyz2cct(wp)) # CIE's illum. E cct ~= 5454
	return cm


def white_balance(img:np.ndarray, wp:List[float], xyz2cam:np.ndarray, cfa_pattern:List[int]) -> np.ndarray:
	"""
		Applies White Balance on a bayer image.

		1. maps scene illumination white point (wp) from XYZ to camera's RGB raw space.
		2. divides each channel by scene's wp, now expressed in camera's RGB raw space (RAW channel multipliers method).
		3. clips the result in the nominal range [0,1].

		Args:
			img (np.ndarray): the image to white balance, this image must be in bayer form.
			wp (list): the image white-point expressed in CIE XYZ_E color space.
			xyz2cam (np.ndarray): color matrix for converting between XYZ and raw RGB.
			cfa_pattern (list): specifies the order of the channels.
		
		Returns:
			the white balanced image in bayer form.

	"""
	wp = np.array(wp).reshape(3,1)
	as_shot_neutral = xyz2cam @ wp
	
	idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
	step2 = 2
	white_balanced_image = np.zeros(img.shape)
	for i, idx in enumerate(idx2by2):
		idx_y = idx[0]
		idx_x = idx[1]
		white_balanced_image[idx_y::step2, idx_x::step2] = img[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]

	white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
	return white_balanced_image


def cam_to_linsrgb(img:np.ndarray, xyz2cam:np.ndarray) -> np.ndarray:
	"""
		cam_E -> XYZ_E -> XYZ_D65 -> linear sRGB_D65
		Args:
			img (np.ndarray): the image to apply color space transform on.
			xyz2cam (np.ndarray): color matrix for converting between XYZ and raw RGB.
	"""

	def row_cm_norm(cm:np.ndarray) -> np.ndarray:
		""" White in camera coords [1, 1, 1] should translate to white [1, 1, 1] in linear sRGB. """
		return cm / np.sum(cm, axis=1, keepdims=True)


	# Bradford adaptation.
	e2d65 = np.array([ [  0.9531874, -0.0265906,  0.0238731 ]
					  ,[ -0.0382467,  1.0288406,  0.0094060 ]
					  ,[  0.0026068, -0.0030332,  1.0892565 ]])

	# XYZ to sRGB, under D65.
	xyz2srgb = np.array([[ 3.2404542, -1.5371385, -0.4985314 ],
						 [-0.9692660,  1.8760108,  0.0415560 ],
						 [ 0.0556434, -0.2040259,  1.0572252 ]])

	cam2srgb = xyz2srgb @ e2d65 @ np.linalg.inv(xyz2cam)
	cam2srgb = row_cm_norm(cam2srgb)
	img = np.squeeze(cam2srgb[None, None, :, :] @ img[:, :, :, None])
	img = np.clip(img, 0.0, 1.0)
	return img


def linsrgb_to_xyz(img:np.ndarray) -> np.ndarray:
	
	# Bradford adaptation.
	d652e = np.array( [[ 1.0503, 0.0271, -0.0233 ]
					 , [ 0.0391, 0.9730, -0.0093 ]
					 , [-0.0024, 0.0026,  0.9181 ]])

	# sRGB to XYZ, under D65.
	srgb2xyz = np.array( [[ 0.4124564, 0.3575761, 0.1804375 ]
						 ,[ 0.2126729, 0.7151522, 0.0721750 ]
						 ,[ 0.0193339, 0.1191920, 0.9503041 ]])
	
	srgb2xyz = d652e @ srgb2xyz
	img = np.squeeze(srgb2xyz[None, None, :, :] @ img[:, :, :, None])
	img = np.clip(img, 0.0, 1.0)
	return img


def gamma_correction(img:np.ndarray) -> np.ndarray:
	img = img ** (1.0 / 2.2)
	return img


def xyz_to_srgb(img):
	""" XYZ_E -> XYZ_D65 -> sRGB_D65 """
	# Bradford adaptation method.
	e2d65 = np.array([ [0.9531874, -0.0265906,  0.0238731]
					  ,[-0.0382467,  1.0288406,  0.0094060]
					  ,[0.0026068, -0.0030332,  1.0892565]])

	# Convertion matrix from XYZ to sRGB, under D65.
	xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
						 [-0.9692660, 1.8760108, 0.0415560],
						 [0.0556434, -0.2040259, 1.0572252]])

	M = xyz2srgb @ e2d65
	img = np.squeeze(M[None, None, :, :] @ img[:, :, :, None])
	img = np.clip(img, 0.0, 1.0)
	img = gamma_correction(img)
	return img


def single_image(img_path:str, debug=False) -> (np.ndarray, dict):

	img = rawpy.imread(img_path).raw_image.copy()
	meta = get_metadata(img_path)

	img = raw_cropping(img, meta['active_area'], meta['default_size'])
	img = normalize(img, meta['black_level'], meta['white_level'])

	wp_e = [1., 1., 1.] # E
	cm1 = meta['cm1'] # A
	cm2 = meta['cm2'] # D65
	xyz2cam = find_opt_cm(wp_e, cm1, cm2)

	cfa_p = meta['cfa_pattern']
	img = white_balance(img, wp_e, xyz2cam, cfa_p)
	img = demosaic(img, cfa_p)
	
	# The whole cam2srgb matrix has to be row normalized
	# that is why we first aquire the linear sRGB image and
	# we then get the XYZ.
	img = cam_to_linsrgb(img, xyz2cam)
	xyz = linsrgb_to_xyz(img)

	if debug:
		srgb = gamma_correction(img)
		plt.imshow((srgb * 255).astype('uint8'))
		plt.show(block=False)

		# Get a figure by converting acquired XYZ img to sRGB.
		plt.figure()
		plt.imshow((xyz_to_srgb(xyz) * 255).astype('uint8'))
		plt.show()
	
	return xyz, meta


if __name__ == '__main__':
	# img_path = "/home/ktzevel/Desktop/old_samples_and_algorithm/manual_dataset/moving_front/IMG_0032.dng"
	img_path = "/home/ktzevel/Desktop/DNG/104TRFLR/IMG_0046.dng"
	single_image(img_path)