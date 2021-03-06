import KratosMultiphysics
import KratosMultiphysics.RomApplication as romapp
from KratosMultiphysics.RomApplication.structural_mechanics_analysis_rom_LOCAL_POD import StructuralMechanicsAnalysisROM

def TrainHROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = StructuralMechanicsAnalysisROM(model,parameters,'EmpiricalCubature')
    simulation.Run()


if __name__ == "__main__":
    TrainHROM()



