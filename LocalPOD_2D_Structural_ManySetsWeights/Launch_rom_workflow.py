from TrainROM import TrainROM
from TrainHROM import TrainHROM
from RunHROM import RunHROM

import numpy as np
from matplotlib import pyplot as plt

def L2_difference(Original, Approximation):
    percentual_diff = (np.linalg.norm( Approximation -  Original ))   /  (np.linalg.norm(Original)) * 100
    return percentual_diff



def check_error():
    FOM = np.load('SnapshotMatrix.npy')
    ROM = np.load('HROM_snapshots.npy')

    L2 = L2_difference(FOM , ROM)
    return L2


if __name__=='__main__':

    errors = []
    number_of_elements = []
    for i in range(1,20):
        TrainROM(i)
        TrainHROM()
        num_elemns = RunHROM()
        number_of_elements.append(num_elemns)
        errors.append(check_error())


    plt.plot(errors)
    plt.title(r'Recons error vs # of clusters', fontsize=10, fontweight='bold')
    plt.xlabel('number_of_clusters')
    plt.ylabel('Percentual L2 error')
    plt.show()


    plt.plot(number_of_elements)
    plt.title(r'#selected HROM elements vs # clusters', fontsize=10, fontweight='bold')
    plt.xlabel('number_of_clusters')
    plt.ylabel('number selected elements')
    plt.show()

    np.save('errors.npy', np.array(errors) )
    np.save('number_of_elements.npy', np.array(number_of_elements) )
