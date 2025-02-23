import os
import argparse
from tqdm import tqdm # pip install tqdm
from scipy.io import loadmat # pip install scipy
import numpy as np # pip install numpy

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--input_dir', type=str, help="The path to the input directory with the .mat files.", required=True)
parser.add_argument('-n', '--var_name', type=str, help="The name of the variable in the .mat file.", required=True)
parser.add_argument('-t', '--var_type', type=str, help="The type of the variable. (double/single)", required=True)
args = parser.parse_args()

input_dir = args.input_dir
var_name = args.var_name
var_type = args.var_type

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
            if ext(f) == 'mat':
                mat = loadmat(os.path.join(root, f))
                if var_type == 'double':
                    a = mat[var_name].astype(np.double)
                elif var_type == 'single':
                    a = mat[var_name].astype(np.single)
                else:
                    raise ValueError('This variable type is not included in the available options.')

                path = os.path.join(root, f.split('.')[0] + '.npy')
                np.save(path, a)
