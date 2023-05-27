import argparse
import numpy as np
from hdf5storage import savemat
from tqdm import tqdm
import os, psutil
process = psutil.Process(os.getpid())
from humanfriendly import format_size

def main(args):
    for file in tqdm(args.files):
        if not os.path.exists(f'{os.path.splitext(file)[0]}.mat'):
            a = np.load(file)
            print(f'{file} opened {format_size(process.memory_info().rss)}')
            d = dict(a)
            print(f'saving to {os.path.splitext(file)[0]}.mat {format_size(process.memory_info().rss)}')
            savemat(f'{os.path.splitext(file)[0]}.mat', d, format = '7.3')
            print(f'{os.path.splitext(file)[0]}.mat saved {format_size(process.memory_info().rss)}')
        else:
            print(f'{os.path.splitext(file)[0]}.mat already saved {format_size(process.memory_info().rss)}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert NPZ files to MAT files')
	parser.add_argument('files', metavar='F', nargs="+", help='NPZ files')
	args = parser.parse_args()
	main(args)