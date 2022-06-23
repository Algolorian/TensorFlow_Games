import numpy as np
from glob import glob

npy_arr = None
bln_arr = None

ls = glob('Compiled_Records/*.npy')
lns = len(ls)
for filename in ls:
    print(ls.index(filename), lns)
    if npy_arr is None:
        npy_arr = np.array([np.load(filename, allow_pickle=True)])
        bln_arr = np.array([np.genfromtxt(f'{filename[:-4]}.csv', dtype='int8')])
    else:
        npy_arr = np.concatenate((npy_arr, np.array([np.load(filename, allow_pickle=True)])))
        bln_arr = np.concatenate((bln_arr, np.array([np.genfromtxt(f'{filename[:-4]}.csv', dtype='int8')])))

# Comment out if not existing
# npy_arr = np.concatenate((npy_arr, np.load('Sample_Files/npy_arr.npy', allow_pickle=True)))
# bln_arr = np.concatenate((bln_arr, np.load('Sample_Files/bln_arr.npy')))


np.save(f"Sample_Files/npy_arr", npy_arr)
np.save(f"Sample_Files/bln_arr", bln_arr)
