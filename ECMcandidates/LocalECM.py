import numpy as np
from empirical_cubature_method_candidates import EmpiricalCubatureMethod

#to save the python numpy array in matlab format
import scipy.io

from matplotlib import pyplot as plt

def local_ecm(number_bases, path_bases):

    #TODO How does order affect the resulting selected elements and weights???
    #sorting first ???

    ElementSelector = EmpiricalCubatureMethod()
    z_i = []
    w_i = []
    z = None
    for i in range(number_bases):
        basis_i = np.load(path_bases + f'{i}.npy' )
        scipy.io.savemat(f'Basis{i}.mat', dict(SnapshotMatrix=basis_i))

        ElementSelector.SetUp(basis_i.T, z)
        ElementSelector.Initialize()
        ElementSelector.Calculate()
        w_i.append(np.squeeze(ElementSelector.w))
        z_i.append(np.squeeze(ElementSelector.z))
        if z is None:
            z = z_i[i]
        else:
            z = np.union1d(z,z_i[i])
        print(z)

    WeightsMatrix = np.zeros(( np.size(z) ,number_bases))

    for i in range(number_bases):
        for j in range(len(z_i[i])):
            index = np.where( z ==   z_i[i][j] )
            if np.array([0]) == index[0]:
                WeightsMatrix[index[0] , i] = w_i[i][j]
            elif not index[0]:
                pass
            else:
                WeightsMatrix[index[0] , i] = w_i[i][j]

    #TODO save the element id's in a list with their corresponding weight

    #1.- Initialize a vector of zeros for each of the selected elements
    #2.- Loop over the elements of the dictionary, and impose the values in
    # the corresponding entry of the vector of weigths
    #3.- Online, call the correct weight by using the Currentcluster() method


    return WeightsMatrix , z

def PlotMatrix(matrix):
    plt.spy(matrix)
    plt.show()
    plt.plot(matrix)
    plt.show()



if __name__=='__main__':
    print('a matrix per cluser')
    WeightsMatrix, z =  local_ecm(6,'./ResidualProjectedOnBasis')
    PlotMatrix(WeightsMatrix)
    np.save('WeightsMatrix.npy',WeightsMatrix )
    np.save('Elementsvector.npy', z)
    print('\na global matrix')
    local_ecm(1,'./SingleMatrixResidual')

