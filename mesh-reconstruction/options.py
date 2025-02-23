import argparse
import json
from easydict import EasyDict

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--rgb_dir', type=str, default='data/acdc_ref_images', help='path to rgb images')
        self.parser.add_argument('--semantics_dir', type=str, default='data/semantic_label_ids', help='path to semantic annotations')
        self.parser.add_argument('--instances_dir', type=str, default='data/instance_label_ids', help='path to instance annotations')
        self.parser.add_argument('--depth_dir', type=str, default='data/depth/unidepth-vit-bilinear', help='path to depth after filter')
        self.parser.add_argument('--convert_to_meters', action='store_true', help='Converts from mm to m optionally.')
        self.parser.add_argument('--normal_dir', type=str, default='data/normals/idisc-trained_on_nyu-acdc-normals', help='path to idisc normal model config file')
        self.parser.add_argument('--out_dir', type=str, default="output", help='Output dir')
        
        # Increase the filter size and the handle_intersect layer will handle rest
        self.parser.add_argument('--uncertain_filter_size', type=int, default=10, help='Size of filter used to find uncertain maps')
        self.parser.add_argument('--uncertain_variance_limit', type=float, default=1e-2, help='Variance filter used to find uncertain maps')

        self.parser.add_argument('--calibration', type=str, default='gopro4', help='Camera intrinsics.')

        self.parser.add_argument('--depth_refine_lr', type=float, default=2e-4, help='Learning rate for depth refinement kernel')
        self.parser.add_argument('--depth_refine_iter', type=float, default=1000, help='Iterations for depth refinement kernel')
        self.parser.add_argument('--depth_refine_l1', type=float, default=1, help='Weight for normal loss')
        self.parser.add_argument('--depth_refine_l2', type=float, default=50, help='Weight for continuity loss')
        self.parser.add_argument('--depth_refine_l3', type=float, default=1, help='Weight for depth loss')

        self.parser.add_argument('--post_process_margin', type=int, default=5, help='Post-processing parameter: margin')

        self.parser.add_argument('--all', action='store_true', help='To run for all available images.')
        self.parser.add_argument('--image_id', type=str, default="GOPR0122_frame_000064", help='For a single image specify an image_id to use. (e.g. GOPR0122_frame_000028, GOPR0351_frame_000504, GOPR0351_frame_000159)')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = EasyDict(vars(self.parser.parse_args()))
        args = self.opt = vars(self.opt)
        self.print(args)
        return self.opt


    def print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def load_depth_config(self):
        with open(self.depth_config, "r") as f:
            config = json.load(f)
        return config

    def load_normal_config(self):
        with open(self.normal_config, "r") as f:
            config = json.load(f)
        return config
