import numpy as np
from empirical_cubature_method_candidates import EmpiricalCubatureMethod

#to save the python numpy array in matlab format
import scipy.io



def local_ecm(number_bases, path_bases):
    #sorting first ???
    ElementSelector = EmpiricalCubatureMethod()
    z = None
    w = []
    for i in range(number_bases):
        basis_i = np.load(path_bases + f'{i}.npy' )
        scipy.io.savemat(f'Basis{i}.mat', dict(SnapshotMatrix=basis_i))

        ElementSelector.SetUp(basis_i.T, z)
        ElementSelector.Initialize()
        ElementSelector.Calculate()
        w.append(np.squeeze(ElementSelector.w))
        z_i = np.squeeze(ElementSelector.z)
        if z is None:
            z = z_i
        else:
            z = np.union1d(z,z_i)
        print(z)






if __name__=='__main__':
    print('a matrix per cluser')
    local_ecm(6,'./ResidualProjectedOnBasis')
    print('\na global matrix')
    local_ecm(1,'./SingleMatrixResidual')

