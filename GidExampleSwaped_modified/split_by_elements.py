import numpy as np
import os
import sys
import pathlib
import pdb


def split_snapshots_by_elements():
    NumberOfSnapshotSubMatrices = 0
    for path in pathlib.Path("./SubMatrices2/").iterdir():
        if path.is_file():
            NumberOfSnapshotSubMatrices += 1

    newpath = r'./NewlySplittedMatricesByElement'

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    NumberOfPartitions = 10

    for j in range(NumberOfPartitions):
        SubMatrixBySnapshots_keep = None
        for i in range(NumberOfSnapshotSubMatrices):
            print(f'extracting matrix from sub-snapthot {i}')
            SubMatrixBySnapshots_erase = np.load(f'./SubMatrices2/SubSnapshot{i}.npy')
            partition_size = int(np.shape(SubMatrixBySnapshots_erase)[0]/NumberOfPartitions)

            if j == NumberOfPartitions-1:
                SubMatrixBySnapshots_erase = SubMatrixBySnapshots_erase[partition_size*j:,:]
            else:
                SubMatrixBySnapshots_erase = SubMatrixBySnapshots_erase[partition_size*j:partition_size*(j+1),:]

            if SubMatrixBySnapshots_keep is None:
                SubMatrixBySnapshots_keep = SubMatrixBySnapshots_erase
            else:
                SubMatrixBySnapshots_keep = np.c_[SubMatrixBySnapshots_keep, SubMatrixBySnapshots_erase]

        np.save(f'./NewlySplittedMatricesByElement/PartitionedSubmatrix{j}',SubMatrixBySnapshots_keep)
        print('the newly created matrix is of shape: ',np.shape(SubMatrixBySnapshots_keep))








if __name__=='__main__':
    split_snapshots_by_elements()



