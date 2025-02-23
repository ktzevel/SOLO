import os
import argparse
from tqdm import tqdm # pip install tqdm
from scipy.io import savemat # pip install scipy
import numpy as np # pip install numpy

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--input_dir', type=str, help="The path to the input directory with the .npy files.", required=True)
parser.add_argument('-n', '--var_name', type=str, help="The name of the variable in the .mat file.", required=True)
args = parser.parse_args()

input_dir = args.input_dir
var_name = args.var_name

if __name__ == "__main__":
    if not os.path.exists(input_dir):
        print(f'{input_dir} does not exist.')
        exit()

    if os.path.isfile(input_dir):
        print(f'{input_dir} is not a directory.')
        exit()

    ext = lambda fn: fn.split('.')[-1]
    for (root, _, files) in os.walk(input_dir):
        for f in tqdm(files):
            if ext(f) == 'npy':
                a = np.load(os.path.join(root, f))
                path = os.path.join(root, f.split('.')[0] + '.mat')
                savemat(path, {var_name: a})
