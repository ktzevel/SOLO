""" 
	These are the calibration parameters of the GoPro4 model in narrow-field mode taken from: https://argus.web.unc.edu/camera-calibration-database/
	The distortion params are not needed since the dataset is captured using the linear FoV mode according to: https://community.gopro.com/s/article/What-is-Linear-Field-Of-View-FOV
"""
gopro4 = {
		  'fx': 1780.0
		, 'fy': 1780.0

		, 'cx': 959.5
		, 'cy': 539.5

		, 'k1': -0.255
		, 'k2': -0.07
		, 'k3': 0.3

		, 't1': 0.0
		, 't2': 0.0
}
