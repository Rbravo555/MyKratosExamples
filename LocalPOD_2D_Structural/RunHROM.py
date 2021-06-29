import KratosMultiphysics
import KratosMultiphysics.RomApplication as romapp
from KratosMultiphysics.RomApplication.structural_mechanics_analysis_rom_LOCAL_POD import StructuralMechanicsAnalysisROM
import numpy as np


import json

class HROM_simulation(StructuralMechanicsAnalysisROM):

    def __init__(self,model,project_parameters):
        super().__init__(model,project_parameters)
        self.time_step_solution_container = []

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        ArrayOfDisplacements = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfDisplacements.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_X, 0))
            ArrayOfDisplacements.append(node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y, 0))
        self.time_step_solution_container.append(ArrayOfDisplacements)


    def EvaluateQuantityOfInterest(self):
       ##############################################################################################
       #     Functions evaluating the QoI of the problem: Snapshot matrix of every time step        #
       #                                                                                            #
       ##############################################################################################
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        return SnapshotMatrix

    def ModifyInitialGeometry(self):
        """Here is the place where the BASIS_ROM and the AUX_ID are imposed to each node"""
        super().ModifyInitialGeometry()
        self.numer_selected_elements = 0
        computing_model_part = self._solver.GetComputingModelPart()
        ## Adding the weights to the corresponding elements
        with open('ElementsAndWeights.json') as f:
            HR_data = json.load(f)
            for key in HR_data["Elements"].keys():
                self.numer_selected_elements+=1
                #lets focus only on the ROM for now, not hyper-reducing...
                #lets use hrom already
                computing_model_part.GetElement(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Elements"][key])
            for key in HR_data["Conditions"].keys():
                self.numer_selected_elements+=1
                computing_model_part.GetCondition(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Conditions"][key])


def RunHROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = HROM_simulation(model,parameters)
    simulation.Run()
    np.save('HROM_snapshots.npy', simulation.EvaluateQuantityOfInterest())
    return simulation.numer_selected_elements


if __name__ == "__main__":
    RunHROM()

