import numpy as np
from matplotlib import pyplot as plt

if __name__=='__main__':
    global_weights_FINAL = np.load('global_weights_FINAL.npy')

    global_weights_naive = np.load('global_weights_naive.npy')

    matrix_of_selected_elements_naive = np.load('matrix_of_selected_elements_naive.npy')

    matrix_of_selected_elements_FINAL = np.load('matrix_of_selected_elements_FINAL.npy')

    sum_of_all_elemental_residuals = np.load('sum_of_all_elemental_residuals.npy')


    naive_solution = matrix_of_selected_elements_naive.T @ global_weights_naive

    Final_solution = matrix_of_selected_elements_FINAL.T @ global_weights_FINAL


    single_ECM = np.load('single_ECM_application.npy')

    plt.plot(naive_solution, 'ro', lineWidth=3, label ='naive_solution')
    plt.plot(Final_solution,'bo' ,lineWidth=3, label ='Final_solution')
    plt.plot(sum_of_all_elemental_residuals,'k' ,lineWidth=3, label ='Exact Solution')
    plt.plot(single_ECM,'go' ,lineWidth=3, label ='Single application')


    plt.legend()
    #plt.yscale('log')
    plt.title('Comparison of Sum of residuals by spatial partition approaches', fontsize=15, fontweight='bold')
    #plt.title('Singular Values for matrix S1', fontweight='bold')
    # plt.xlabel('entry number in the diagonal of the matrix sigma')
    # plt.ylabel('singular value magnitude (log scale)')
    plt.show()


