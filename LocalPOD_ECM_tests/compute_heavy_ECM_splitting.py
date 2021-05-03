import numpy as np
from empirical_cubature_method import EmpiricalCubatureMethod
import pdb

if __name__== '__main__':
    NumberOfSnapshotSubMatrices = 10
    ElementSelector = EmpiricalCubatureMethod()
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
        TotalElemsSubdomains += np.shape(SubMatrixBySnapshots_erase)[0]
        ElementSelector.SetUp(SubMatrixBySnapshots_erase, np.ones(np.shape(SubMatrixBySnapshots_erase)[0]))
        ElementSelector.Initialize()
        print('Computing ECM... \n')
        ElementSelector.Calculate()

        weights = np.squeeze(ElementSelector.w)
        weights = np.reshape(weights,[np.size(weights),1])
        elements = np.squeeze(ElementSelector.z)

        #pdb.set_trace()

        if elements_for_saving is None:
            elements_for_saving = np.reshape(elements, (1,-1))
            element_to_add_next_submatrix = TotalElemsSubdomains
        else:
            elements_for_saving = np.c_[elements_for_saving,  np.reshape(elements+ element_to_add_next_submatrix, (1,-1))]
            element_to_add_next_submatrix = TotalElemsSubdomains


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
        #pdb.set_trace()


    #pdb.set_trace()

    np.save('selected_elements', elements_for_saving)
    np.save('Matrix_of_selected_elements', Matrix_Keep)
    np.save('selected_weights', np.squeeze(global_weights))

    # ElementSelector.SetUp(Matrix_Keep, np.squeeze(global_weights))
    # ElementSelector.Initialize()
    # ElementSelector.Calculate()

    # weights = np.squeeze(ElementSelector.w)
    # weights = np.reshape(weights,[np.size(weights),1])
    # elements = np.squeeze(ElementSelector.z)

    # pdb.set_trace()
    # for i in range(np.size(elements)):
    #     print('it was: ',elements[i])
    #     print('it is: ',ElementMapping[elements[i]])
    #     elements[i] = ElementMapping[elements[i]]

    # OriginalNumberOfElements = 120950
    # ModelPartName = "3D_already_in_gid_coarse"
    # ElementSelector.WriteSelectedElements(weights, elements, OriginalNumberOfElements)

    # ElementSelector._CreateHyperReducedModelPart(self, ModelPartName)
    # #taking into account the partition bias to the selected elements
