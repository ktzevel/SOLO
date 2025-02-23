% No need to do the processing, take in depth map directly
function d = depth_postprocessing_dual_bilateral_filter(d, image, labels,...
    L, depth_postprocessing_parameters, varargin)
%TRANSMITTANCE_POSTPROCESSING_DUAL_BILATERAL_FILTER  Postprocess an already
%complete transmittance map with a dual-range cross-bilateral filter using
%semantics and color as references.

I_CIELAB = RGB_to_CIELAB_unit_range(image);
intensity_min = shiftdim(min(min(I_CIELAB, [], 1), [], 2)).';
intensity_max = shiftdim(max(max(I_CIELAB, [], 1), [], 2)).';

% Read parameters for dual-range color cross-bilateral filter.
sigma_spatial = depth_postprocessing_parameters.sigma_spatial;
sigma_intensity = depth_postprocessing_parameters.sigma_intensity;
kernel_radius_in_std =...
    depth_postprocessing_parameters.kernel_radius_in_std;
lambda = depth_postprocessing_parameters.lambda;

% Set the rest parameters to their recommended values.
sampling_spatial = sigma_spatial;
sampling_intensity = sigma_intensity;

% Postprocess by filtering.
d = dual_range_depth_cross_bilateral_filter_with_bilateral_grids(d, labels,...
    I_CIELAB, 1, L, intensity_min, intensity_max, sigma_spatial,...
    sampling_spatial, sigma_intensity, sampling_intensity,...
    kernel_radius_in_std, lambda);

end

