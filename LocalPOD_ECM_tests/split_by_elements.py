import numpy as np
import os
import sys
import pathlib
import pdb


if __name__=='__main__':

    SingleSnapshots = np.load('Snapshots_Of_residuals.npy')
    print(SingleSnapshots.shape)


    newpath = r'./NewlySplittedMatricesByElement'

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    NumberOfPartitions = 10

    partition_size = int(np.shape(SingleSnapshots)[0]/NumberOfPartitions)
    for j in range(NumberOfPartitions):
        SubMatrixBySnapshots_keep = None
        print(f'extracting matrix from sub-snapthot {j}')

        if j == NumberOfPartitions-1:
            SAVE_IT = SingleSnapshots[partition_size*j:,:]
        else:
            SAVE_IT = SingleSnapshots[partition_size*j:partition_size*(j+1),:]

        np.save(f'./NewlySplittedMatricesByElement/PartitionedSubmatrix{j}',SAVE_IT)
        print('the newly created matrix is of shape: ',np.shape(SAVE_IT))







