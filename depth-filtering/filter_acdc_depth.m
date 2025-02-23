function filter_acdc_depth(img_dir, sem_dir, depth_dir, out_dir)
%
% INPUTS:
%
%   -|img_dir|: char The path to the rgb images directory.
%   -|sem_dir|: char The path to the semantic annotations directory of the corresponding rgb image.
%   -|depth_dir|: char The path to the depth directory of the corresponding rgb image.
%   -|out_dir|: char The path to the output directory where the resulted depth will be stored.
%
% ------------------------------------------------------------------------------

% Add required paths.

curr_path = mfilename('fullpath');
curr_dir = fileparts(curr_path);

addpath(fullfile(curr_dir, 'utilities'));
addpath_relative_to_caller(curr_path, 'Fog_simulation');
addpath_relative_to_caller(curr_path, 'Depth_processing');
addpath_relative_to_caller(curr_path, 'Dark_channel_prior');
addpath_relative_to_caller(curr_path, 'Instance-level_semantic_segmentation');
addpath_relative_to_caller(curr_path, 'Color_transformations');
addpath_relative_to_caller(curr_path, fullfile('external', 'SLIC_mex'));

% ------------------------------------------------------------------------------

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

files = dir(img_dir);
n = numel(files);
for i = 1:numel(files)
    if files(i).isdir
        continue;
    end

    img_fn = files(i).name;
    sem_fn = img_fn;
    depth_fn = strrep(img_fn, ".png", ".mat"); 

    img_path = fullfile(img_dir, img_fn);
    sem_path = fullfile(sem_dir, sem_fn);
    depth_path = fullfile(depth_dir, depth_fn);

    fn = strrep(img_fn, ".png", ".mat"); 
    out_path = fullfile(out_dir, fn);
    if isfile(out_path)
        continue;
    end

    % Bring original, clear left image to double precision for subsequent computations.
    image = im2double(imread(img_path));

    load(depth_path, 'depth');
    depth = im2double(depth);
    max_depth = max(max(depth));
    depth = (depth / max_depth); % normalize

    % Define DBF parameters.
    params = struct('sigma_spatial', 5 ...
                  , 'sigma_intensity', 0.1 ...
                  , 'kernel_radius_in_std', 1 ...
                  , 'lambda', 0 ...
                  );

    % Cast the GT labels to canonical range.
    sem_ids = imread(sem_path);
    [sem_lbls, L] = instance_ids_to_label_image_plain_cityscapes(sem_ids);
    res_depth = depth_postprocessing_dual_bilateral_filter(depth, image, sem_lbls, L, params);
    
    % scale up to metric space again and store it.
    res_depth = res_depth * max_depth;
    save(out_path, "res_depth");

end
