import numpy as np
from empirical_cubature_method import EmpiricalCubatureMethod
import pdb

if __name__== '__main__':


    ElementSelector = EmpiricalCubatureMethod()





    #Single_application_of_ECM
    kkk = np.load('Snapshots_Of_residuals.npy')
    ElementSelector.SetUp(kkk, np.ones(np.shape(kkk)[0]))
    ElementSelector.Initialize()
    print('Computing ECM... \n')
    ElementSelector.Calculate()
    weights = np.squeeze(ElementSelector.w)
    weights = np.reshape(weights,[np.size(weights),1])
    elements = np.squeeze(ElementSelector.z)
    Matrix_Single = kkk[elements,:]
    Vector_Single = Matrix_Single.T @ weights
    np.save('single_ECM_application.npy', Vector_Single)





    NumberOfSnapshotSubMatrices = 10
    Matrix_Keep = None
    TotalElemsSubdomains = 0
    AddToElems = 0
    AddToIndex = 0
    global_weights = None

    ElementMapping = {}
    elements_for_saving = None
    for i in range(NumberOfSnapshotSubMatrices):
        print(f'Selecting elements from the matrix {i}')
        print('loading matrix... \n')
        SubMatrixBySnapshots_erase = np.load(f'./PartitionedSubmatrix{i}.npy')
        if i==0:
            gobal_sum_vector = SubMatrixBySnapshots_erase.T @ np.ones((np.shape(SubMatrixBySnapshots_erase)[0],1)) #this vector contains the sum of the residual over all the elements
        else:
            gobal_sum_vector+= SubMatrixBySnapshots_erase.T @ np.ones((np.shape(SubMatrixBySnapshots_erase)[0],1)) #this vector contains the sum of the residual over all the elements

        TotalElemsSubdomains = np.shape(SubMatrixBySnapshots_erase)[0]
        ElementSelector.SetUp(SubMatrixBySnapshots_erase, np.ones(np.shape(SubMatrixBySnapshots_erase)[0]))
        ElementSelector.Initialize()
        print('Computing ECM... \n')
        ElementSelector.Calculate()

        weights = np.squeeze(ElementSelector.w)
        weights = np.reshape(weights,[np.size(weights),1])
        elements = np.squeeze(ElementSelector.z)

        for i in range(np.size(elements)):
            ElementMapping.update( {i+AddToIndex:elements[i]+AddToElems})
        AddToIndex+= np.size(elements)
        AddToElems+= TotalElemsSubdomains

        if Matrix_Keep is None:
            Matrix_Keep = SubMatrixBySnapshots_erase[elements,:]
        else:
            Matrix_Keep = np.r_[Matrix_Keep, SubMatrixBySnapshots_erase[elements,:]]

        if global_weights is None:
            global_weights = weights
        else:
            global_weights = np.r_[global_weights, weights]


    np.save('global_weights_naive.npy', global_weights)
    np.save('matrix_of_selected_elements_naive.npy', Matrix_Keep)
    np.save('sum_of_all_elemental_residuals.npy', gobal_sum_vector)




    #global_weights = np.load('global_weights_naive.npy')
    #Matrix_Keep = np.load('matrix_of_selected_elements_naive.npy')

    ElementSelector.SetUp(Matrix_Keep, np.squeeze(global_weights))
    ElementSelector.Initialize()
    ElementSelector.Calculate()

    weights = np.squeeze(ElementSelector.w)
    weights = np.reshape(weights,[np.size(weights),1])
    elements = np.squeeze(ElementSelector.z)

    FinalMatrix = Matrix_Keep[elements,:]
    FinalWeights = weights



    print(elements)

    for i in range(np.size(elements)):
        print('it was: ',elements[i])
        print('it is: ',ElementMapping[elements[i]])
        elements[i] = ElementMapping[elements[i]]


    pdb.set_trace()

    np.save('matrix_of_selected_elements_FINAL.npy', FinalMatrix)
    np.save('global_weights_FINAL.npy', FinalWeights)





    OriginalNumberOfElements = 120950
    ModelPartName = "BarForClustering2"
    ElementSelector.WriteSelectedElements(ElementSelector.w, elements, OriginalNumberOfElements)

    #ElementSelector._CreateHyperReducedModelPart(ModelPartName)
    #taking into account the partition bias to the selected elements

