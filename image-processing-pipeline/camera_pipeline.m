    % --------------------------------------------------------
% Processing DNG images in Matlab
% Based on:
% 1. https://rcsumner.net/raw_guide/RAWguide.pdf
% 2. https://doi.org/10.1117/1.OE.59.11.110801
%
% --------------------------------------------------------

clear;
clc;

warning off MATLAB:tifflib:TIFFReadDirectory:libraryWarning
% filename = '/home/ktzevel/Desktop/old_samples_and_algorithm/manual_dataset/moving_front/IMG_0032.dng'; % Put file name here
% filename = '/home/ktzevel/Desktop/old_samples_and_algorithm/manual_dataset/traffic_light_R/IMG_0028.dng'; % Put file name here
filename = '/home/ktzevel/Desktop/DNG/104TRFLR/IMG_0046.dng';

% --------------------------------------------------------
% Fetching the raw image.
% --------------------------------------------------------
t = Tiff(filename,'r');
offsets = getTag(t,'SubIFD');
setSubDirectory(t, offsets(1));
raw = read(t); % Create variable ’raw’, the Bayer CFA data
close(t);

% Fetching metadata.
meta_info = imfinfo(filename);

% --------------------------------------------------------
% Crop raw image according to active_area and DefaultCrop tags.
% --------------------------------------------------------
x_origin = meta_info.SubIFDs{1}.ActiveArea(2)+1; % +1 due to MATLAB indexing
width = meta_info.SubIFDs{1}.DefaultCropSize(1);
y_origin = meta_info.SubIFDs{1}.ActiveArea(1)+1;
height = meta_info.SubIFDs{1}.DefaultCropSize(2);
raw = double(raw( y_origin:y_origin + height - 1, x_origin:x_origin + width - 1 ));

% --------------------------------------------------------
% Linearization
% --------------------------------------------------------
% Some manifactures use non-linear compress schemes for storing raw images (e.g. Nikon)
% Canon cameras do not do that.
if isfield(meta_info.SubIFDs{1},'LinearizationTable')
	ltab=meta_info.SubIFDs{1}.LinearizationTable;
	raw = ltab(raw+1);
end

% --------------------------------------------------------
% Normalization
% --------------------------------------------------------
black = meta_info.SubIFDs{1}.BlackLevel(1);
saturation = meta_info.SubIFDs{1}.WhiteLevel;
lin_bayer = ( raw - black ) / ( saturation - black );

% Clips the image in [0,1]
lin_bayer = max(0, min(lin_bayer, 1));

% --------------------------------------------------------
% White balance
% --------------------------------------------------------
% Uses camera's AWB values.
%wb_multipliers = (meta_info.AsShotNeutral).^-1;

% Equal-energy radiator, [1.000, 1.000, 1.000].
wp_e = whitepoint("e")';

% Color-matrices go from XYZ to camera color basis.
cm1 = meta_info.ColorMatrix1; % CIE's A
cm1 = reshape(cm1, 3, 3)'; % These are stored as 1x9 in a C row-wise manner.

cm2 = meta_info.ColorMatrix2; % CIE's D65
cm2 = reshape(cm2, 3, 3)';

% Typically normalize color-matrices s.t. chracterization illuminant just saturates
% in the camera space. 
cm1 = wp_cm_norm(cm1, wp_e);
cm2 = wp_cm_norm(cm2, wp_e);

cct_e = 5454;
xyz2cam = optimize_cm(cm1, cm2, cct_e); % optimizing for illuminant E.

wp_e = xyz2cam * wp_e; % converting it to camera space coordinates.
wb_multipliers = wp_e' .^ -1;

wb_multipliers = wb_multipliers / wb_multipliers(2);
mask = wbmask(size(lin_bayer, 1), size(lin_bayer, 2), wb_multipliers, 'rggb');
balanced_bayer = lin_bayer .* mask;

% Clips the image in [0,1]
balanced_bayer = max(0, min(balanced_bayer, 1));

% --------------------------------------------------------
% Demosaicing
% --------------------------------------------------------
temp = balanced_bayer / max(balanced_bayer(:)) * 65535;
dem = double(demosaic(uint16(temp),'rggb')) / 65535;

% --------------------------------------------------------
% Color space transform
% --------------------------------------------------------

e2d65 = [ 0.9531874 -0.0265906  0.0238731; ...
		 -0.0382467  1.0288406  0.0094060; ...
 		  0.0026068 -0.0030332  1.0892565 ];

xyz2srgb = [3.2404542 -1.5371385 -0.4985314; ...
			-0.9692660  1.8760108  0.0415560; ...
			0.0556434 -0.2040259  1.0572252];

cam2srgb = xyz2srgb * e2d65 / xyz2cam;

% row-normalization (taken from dcraw source code).
% A white point (1.0, 1.0, 1.0) in the camera space should result
% in a white point in the sRGB space.
cam2srgb = cam2srgb ./ repmat(sum(cam2srgb , 2), 1, 3);

lin_srgb = apply_cmatrix(dem, cam2srgb);
lin_srgb = max(0, min(lin_srgb, 1)); % Clips to [0,1]

% xyz = apply_cmatrix(lin_srgb, inv(e2d65) * inv(xyz2srgb));

% Gamma correction.
nl_srgb = lin_srgb .^ (1 / 2.2);


imshow(nl_srgb)

% f = figure;
% imshow(apply_cmatrix(xyz, xyz2srgb).^ (1 / 2.2))



% -------------- Function definitions follow: -------------- %

function cm = wp_cm_norm(cm, wp)

	maxval = max(cm * wp);
	cm = cm / maxval;
end

function cm = optimize_cm(cm1, cm2, temp)
	% Optimizes color matrix for illuminant with the specified cct temperature.

	cct1 = 2856;
	cct2 = 6504;

	if temp <= cct1
		g = 1.0;
	elseif temp >= cct2
		g = 0.0;
	else
		tempinv = 1.0 / temp;
		cct1inv = 1.0 / cct1;
		cct2inv = 1.0 / cct2;
		g = (tempinv - cct2inv) / (cct1inv - cct2inv);
	end

	if g >= 1.0
		cm = cm1;
		return
	end

	if  g <= 0.0
		cm = cm2;
		return
	end

	cm = g * cm1 + (1.0 - g) * cm2;
end


function colormask = wbmask(m, n, wbmults, align)
	% COLORMASK = wbmask(M,N,WBMULTS,ALIGN)
	%
	% Makes a white-balance multiplicative mask for an image of size m-by-n
	% with RGB while balance multipliers WBMULTS = [R_scale G_scale B_scale].
	% ALIGN is string indicating Bayer arrangement: ’rggb’,’gbrg’,’grbg’,’bggr’
	colormask = wbmults(2)*ones(m,n); %Initialize to all green values
	switch align
		case 'rggb'
			colormask(1:2:end, 1:2:end) = wbmults(1); %r
			colormask(2:2:end, 2:2:end) = wbmults(3); %b
		case 'bggr'
			colormask(2:2:end, 2:2:end) = wbmults(1); %r
			colormask(1:2:end, 1:2:end) = wbmults(3); %b
		case 'grbg'
			colormask(1:2:end, 2:2:end) = wbmults(1); %r
			colormask(1:2:end, 2:2:end) = wbmults(3); %b
		case 'gbrg'
			colormask(2:2:end, 1:2:end) = wbmults(1); %r
			colormask(1:2:end, 2:2:end) = wbmults(3); %b
	end
end

function corrected = apply_cmatrix(im, cm)
	% CORRECTED = apply_cmatrix(IM, CM)
	%
	% Applies CM to RGB input IM. Finds the appropriate weighting of the
	% old color planes to form the new color planes, equivalent to but much
	% more efficient than applying a matrix transformation to each pixel.
	if size(im, 3) ~= 3
		error('Apply CM to RGB image only.')
	end

	r = cm(1,1) * im(:,:,1) + cm(1,2) * im(:,:,2) + cm(1,3) * im(:,:,3);
	g = cm(2,1) * im(:,:,1) + cm(2,2) * im(:,:,2) + cm(2,3) * im(:,:,3);
	b = cm(3,1) * im(:,:,1) + cm(3,2) * im(:,:,2) + cm(3,3) * im(:,:,3);

	corrected = cat(3, r, g, b);
end