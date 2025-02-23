import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="This script converts semantic level annotations to instance ones.")
parser.add_argument('sem_dir', type=str, help="the path to semantics directory.")
parser.add_argument('out_dir', type=str, help="the path to the output directory.")
args = parser.parse_args()
	
sem_dir = args.sem_dir

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

for root, _, files in os.walk(sem_dir):
    for name in tqdm(files):
        if not name.endswith('.png'):
            continue

        img_path = os.path.join(root, name)
        semantics = np.array(Image.open(img_path))
        instances, _ = label(semantics)

        fn = os.path.basename(img_path)
        out_path = os.path.join(out_dir, fn)
        Image.fromarray(instances).save(out_path)
