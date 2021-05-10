import KratosMultiphysics
import KratosMultiphysics.RomApplication as romapp
from KratosMultiphysics.RomApplication.structural_mechanics_analysis_rom_LOCAL_POD import StructuralMechanicsAnalysisROM
import numpy as np


import json

class HROM_simulation(StructuralMechanicsAnalysisROM):

    def ModifyInitialGeometry(self):
        """Here is the place where the BASIS_ROM and the AUX_ID are imposed to each node"""
        super().ModifyInitialGeometry()
        computing_model_part = self._solver.GetComputingModelPart()
        # ## Adding the weights to the corresponding elements
        # with open('ElementsAndWeights.json') as f:
        #     HR_data = json.load(f)
        #     for key in HR_data["Elements"].keys():
        #         computing_model_part.GetElement(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Elements"][key])
        #     for key in HR_data["Conditions"].keys():
        #         computing_model_part.GetCondition(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Conditions"][key])
        WeightsMatrix = np.load('WeightsMatrix.npy')
        ElementsVector = np.load('Elementsvector.npy')
        elemental_vector = KratosMultiphysics.Vector(6)
        for i in range(WeightsMatrix.shape[0]):
            for j in range(6):
                elemental_vector[j] = WeightsMatrix[i,j]
            if ElementsVector[i] + 1  < 800:
                computing_model_part.GetElement(int( ElementsVector[i])+1).SetValue(romapp.HROM_WEIGHT, elemental_vector  )
            else:
                computing_model_part.GetCondition(int( ElementsVector[i] - 800)+1).SetValue(romapp.HROM_WEIGHT, elemental_vector )





def RunHROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = HROM_simulation(model,parameters)
    simulation.Run()


if __name__ == "__main__":
    RunHROM()

