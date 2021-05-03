import numpy as np

if __name__=='__main__':
    for i in range(10):
        if i==0:
            reconstructed_original = np.load(f'PartitionedSubmatrix{i}.npy')
        else:
            reconstructed_original = np.r_[reconstructed_original, np.load(f'PartitionedSubmatrix{i}.npy')]

    print('the new shape is: ', np.shape(reconstructed_original))
    np.save('reconstructed_residual_projected_1e-4.npy', reconstructed_original)