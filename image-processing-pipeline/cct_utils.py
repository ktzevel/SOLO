"""
	Based on:
	- https://github.com/SamsungLabs/day-to-night
	- https://android.googlesource.com/platform/external/dng_sdk/+/refs/heads/main (Adobe DNG SDK)
	- http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_T.html
"""

import numpy as np
import sys


def dot(x, y):
	return np.sum(x * y, axis=-1)


def norm(x):
	return np.sqrt(dot(x, x))


def interpolate_cst(cm1, cm2, temp):
	""" Interpolates color matrix.

		Args:
			cm1 (np.ndarray): A 3x3 color matrix.
			cm2 (np.ndarray): A 3x3 color matrix.
			temp (float): The estimated color correlated temperature of as_shot_neutral.
		
		Returns:
			cm (np.ndarray): A 3x3 transformation matrix from XYZ to camera color space (raw-RGB).
	"""
	cct1 = 2856 # Standard Light A
	cct2 = 6504 # D65

	if temp <= cct1:
		g = 1.0
	elif temp >= cct2:
		g = 0.0
	else:
		tempinv = 1.0 / temp
		cct1inv = 1.0 / cct1
		cct2inv = 1.0 / cct2
		g = (tempinv - cct2inv) / (cct1inv - cct2inv)
	
	if g >= 1.0:
		return cm1

	if g <= 0.0:
		return cm2
	
	return g * cm1 + (1.0 - g) * cm2


def xyz2cct(xyz):
	"""
		Args:
			xyz (list): The xyz coordinates in CIE XYZ (normalized or not).
		
		Returns:
			cct (float): The estimated correlated color temperature.
		
		Description:
			Robertson's method: XYZ to Correlated Color Temperature.
			https://en.wikipedia.org/wiki/Correlated_color_temperature#Robertson's%20method
	"""

	rt = [  # /* reciprocal temperature (K) */
		  sys.float_info.min , 10.0e-6 , 20.0e-6 , 30.0e-6 , 40.0e-6 , 50.0e-6 , 60.0e-6 , 70.0e-6
		, 80.0e-6 , 90.0e-6 , 100.0e-6 , 125.0e-6 , 150.0e-6 , 175.0e-6 , 200.0e-6 , 225.0e-6
		, 250.0e-6 , 275.0e-6 , 300.0e-6 , 325.0e-6 , 350.0e-6 , 375.0e-6 , 400.0e-6 , 425.0e-6
		, 450.0e-6 , 475.0e-6 , 500.0e-6 , 525.0e-6 , 550.0e-6 , 575.0e-6 , 600.0e-6 ]

	uvt = [
		[0.18006, 0.26352, -0.24341], [0.18066, 0.26589, -0.25479], [0.18133, 0.26846, -0.26876],
		[0.18208, 0.27119, -0.28539], [0.18293, 0.27407, -0.30470], [0.18388, 0.27709, -0.32675],
		[0.18494, 0.28021, -0.35156], [0.18611, 0.28342, -0.37915], [0.18740, 0.28668, -0.40955],
		[0.18880, 0.28997, -0.44278], [0.19032, 0.29326, -0.47888], [0.19462, 0.30141, -0.58204],
		[0.19962, 0.30921, -0.70471], [0.20525, 0.31647, -0.84901], [0.21142, 0.32312, -1.0182],
		[0.21807, 0.32909, -1.2168], [0.22511, 0.33439, -1.4512], [0.23247, 0.33904, -1.7298],
		[0.24010, 0.34308, -2.0637], [0.24792, 0.34655, -2.4681], [0.25591, 0.34951, -2.9641],
		[0.26400, 0.35200, -3.5814], [0.27218, 0.35407, -4.3633], [0.28039, 0.35577, -5.3762],
		[0.28863, 0.35714, -6.7262], [0.29685, 0.35823, -8.5955], [0.30505, 0.35907, -11.324],
		[0.31320, 0.35968, -15.628], [0.32129, 0.36011, -23.325], [0.32931, 0.36038, -40.770],
		[0.33724, 0.36051, -116.45]
	]
	
	# LERP(a,b,c) = linear interpolation macro, is 'a' when c == 0.0 and 'b' when c == 1.0 
	lerp = lambda a, b, c: (b - a) * c + a

	if (xyz[0] < 1.0e-20) and (xyz[1] < 1.0e-20) and (xyz[2] < 1.0e-20):
		return -1  # /* protect against possible divide-by-zero failure */

	us = (4.0 * xyz[0]) / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2])
	vs = (6.0 * xyz[1]) / (xyz[0] + 15.0 * xyz[1] + 3.0 * xyz[2])

	dm = 0.0
	di = 0
	i = 0
	N = len(rt)
	for i in range(N):
		di = (vs - uvt[i][1]) - uvt[i][2] * (us - uvt[i][0])
		if (i > 0) and (((di < 0.0) and (dm >= 0.0)) or ((di >= 0.0) and (dm < 0.0))):
			break  # /* found lines bounding (us, vs) : i-1 and i */
		dm = di

	if i == N:
		# /* bad XYZ input, color temp would be less than minimum of 1666.7 degrees, or too far towards blue */
		return -1

	di = di / np.sqrt(1.0 + uvt[i][2] * uvt[i][2])
	dm = dm / np.sqrt(1.0 + uvt[i - 1][2] * uvt[i - 1][2])
	p = dm / (dm - di)  # /* p = interpolation parameter, 0.0 : i-1, 1.0 : i */
	p = 1.0 / (lerp(rt[i - 1], rt[i], p))
	cct = p
	return cct  # /* success */