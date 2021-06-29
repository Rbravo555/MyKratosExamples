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

    errors_ours = []
    errors_farhats = []
    number_of_elements_ours = []
    number_of_elements_farhats = []
    number_of_clusters = []

    for i in range(1,30):
        number_of_clusters.append(i)
        for j in range(2):
            if j==0:
                TrainROM(Number_Of_Clusters=i, run_simulations = False,  Farhats = True, Ours = False)
                TrainHROM()
                num_elemns_farhats = RunHROM()
                number_of_elements_farhats.append(num_elemns_farhats)
                errors_farhats.append(check_error())
            else:
                TrainROM(Number_Of_Clusters=i, run_simulations = False,  Farhats = False, Ours = True)
                TrainHROM()
                num_elemns_ours = RunHROM()
                number_of_elements_ours.append(num_elemns_ours)
                errors_ours.append(check_error())



    plt.plot(number_of_clusters, errors_farhats, label = 'Farhats')
    plt.plot(number_of_clusters, errors_ours, label = 'Ours')
    plt.legend()
    plt.title(r'Recons error vs # of clusters', fontsize=10, fontweight='bold')
    plt.xlabel('number_of_clusters')
    plt.ylabel('Percentual L2 error')
    plt.show()


    plt.plot(number_of_clusters, number_of_elements_farhats, label = 'Farhats')
    plt.plot(number_of_clusters, number_of_elements_ours, label = 'Ours')
    plt.legend()
    plt.title(r'#selected HROM elements vs # clusters', fontsize=10, fontweight='bold')
    plt.xlabel('number_of_clusters')
    plt.ylabel('number selected elements')
    plt.show()

    # np.save('errors.npy', np.array(errors) )
    # np.save('number_of_elements.npy', np.array(number_of_elements) )

