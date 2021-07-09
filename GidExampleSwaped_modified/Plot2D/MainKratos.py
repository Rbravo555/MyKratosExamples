import KratosMultiphysics
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
import KratosMultiphysics.RomApplication as romapp
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
from KratosMultiphysics.RomApplication.fluid_dynamics_analysis_rom import FluidDynamicsAnalysisROM
from matplotlib import pyplot as plt
import json

import pdb
import numpy as np
import sys
import time
import os



class CFD(FluidDynamicsAnalysis):

    def __init__(self,model,project_parameters,flush_frequency=10.0):
        super().__init__(model,project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()
        self.time_step_solution_container = []
        self.control_point_velocity_norm = []
        self.control_point_id = 1341

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.parallel_type == "OpenMP":
            now = time.time()
            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now
        ArrayOfResults = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE, 0))
        self.time_step_solution_container.append(ArrayOfResults)


        vel_norm = 0
        for vel_comp in self._GetSolver().GetComputingModelPart().GetNode(self.control_point_id).GetSolutionStepValue(KratosMultiphysics.VELOCITY, 0):
            vel_norm += vel_comp
        vel_norm = vel_norm/3
        self.control_point_velocity_norm.append(vel_norm)



    def GetSnapshotsMatrix(self):
        ### Building the Snapshot matrix ####
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        self.time_step_solution_container = []
        return SnapshotMatrix

    def GetControlPointVelocityNormHistory(self):
        return np.array(self.control_point_velocity_norm)












class CFD_ROM(FluidDynamicsAnalysisROM):

    def __init__(self,model,project_parameters,flush_frequency=10.0):
        super().__init__(model,project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()
        self.time_step_solution_container = []
        self.control_point_velocity_norm = []
        self.control_point_id = 1341

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.parallel_type == "OpenMP":
            now = time.time()
            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now
        ArrayOfResults = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE, 0))
        self.time_step_solution_container.append(ArrayOfResults)

        vel_norm = 0
        for vel_comp in self._GetSolver().GetComputingModelPart().GetNode(self.control_point_id).GetSolutionStepValue(KratosMultiphysics.VELOCITY, 0):
            vel_norm += vel_comp
        vel_norm = vel_norm/3
        self.control_point_velocity_norm.append(vel_norm)



    def GetControlPointVelocityNormHistory(self):
        return np.array(self.control_point_velocity_norm)


    def GetSnapshotsMatrix(self):
        ### Building the Snapshot matrix ####
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        self.time_step_solution_container = []
        return SnapshotMatrix


















class CFD_HROM(FluidDynamicsAnalysisROM):

    def __init__(self,model,project_parameters,flush_frequency=10.0):
        super().__init__(model,project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()
        self.time_step_solution_container = []
        self.control_point_velocity_norm = []
        self.control_point_id = 1341

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.parallel_type == "OpenMP":
            now = time.time()
            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now
        ArrayOfResults = []
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_X, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, 0))
            ArrayOfResults.append(node.GetSolutionStepValue(KratosMultiphysics.PRESSURE, 0))
        self.time_step_solution_container.append(ArrayOfResults)

        vel_norm = 0
        for vel_comp in self._GetSolver().GetComputingModelPart().GetNode(self.control_point_id).GetSolutionStepValue(KratosMultiphysics.VELOCITY, 0):
            vel_norm += vel_comp
        vel_norm = vel_norm/3
        self.control_point_velocity_norm.append(vel_norm)



    def GetControlPointVelocityNormHistory(self):
        return np.array(self.control_point_velocity_norm)


    def GetSnapshotsMatrix(self):
        ### Building the Snapshot matrix ####
        SnapshotMatrix = np.zeros((len(self.time_step_solution_container[0]), len(self.time_step_solution_container)))
        for i in range(len(self.time_step_solution_container)):
            Snapshot_i= np.array(self.time_step_solution_container[i])
            SnapshotMatrix[:,i] = Snapshot_i.transpose()
        self.time_step_solution_container = []
        return SnapshotMatrix




    def ModifyAfterSolverInitialize(self):
        super().ModifyAfterSolverInitialize()

        pdb.set_trace()
        computing_model_part = self._solver.GetComputingModelPart()
        ## Adding the weights to the corresponding elements
        with open('ElementsAndWeights.json') as f:
            HR_data = json.load(f)
            for key in HR_data["Elements"].keys():
                computing_model_part.GetElement(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Elements"][key])
            for key in HR_data["Conditions"].keys():
                computing_model_part.GetCondition(int(key)+1).SetValue(romapp.HROM_WEIGHT, HR_data["Conditions"][key])





















def L2_difference(Original, Approximation):
    percentual_diff = (np.linalg.norm( Approximation -  Original ))   /  (np.linalg.norm(Original)) * 100
    return percentual_diff














def TrainROM(launch_simulation = True, svd_tolerance = 1e-4):
    if launch_simulation:
        with open("ProjectParameters.json",'r') as parameter_file:
            parameters = KratosMultiphysics.Parameters(parameter_file.read())
        model = KratosMultiphysics.Model()
        simulation = CFD(model,parameters)
        #print(simulation._GetSolver().settings)
        #print(simulation.project_parameters)
        simulation.Run()
        SnapshotsMatrix = simulation.GetSnapshotsMatrix()
        np.save('SnapshotsMatrix.npy', SnapshotsMatrix)
        np.save('control_point_vel_FOM.npy',simulation.GetControlPointVelocityNormHistory())
    else:
        SnapshotsMatrix = np.load('SnapshotsMatrix.npy')

    #compute the SVD
    u, s, _, _ = RandomizedSingularValueDecomposition().Calculate(SnapshotsMatrix, svd_tolerance)
    #u,s,v= np.linalg.svd(SnapshotsMatrix, full_matrices=False) #for consevers energy: np.sum(s[:21])/np.sum(s)

    #pdb.set_trace()

    ## Plotting singular values  ###
    plt.plot( range(0,len(s)), np.log(s), marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.title('Singular Values')
    plt.show()

    ### Saving the nodal basis ###  (Need to make this more robust, hard coded here)
    basis_POD={"rom_settings":{},"nodal_modes":{}}
    basis_POD["rom_settings"]["nodal_unknowns"] = ["VELOCITY_X","VELOCITY_Y","PRESSURE"]
    basis_POD["rom_settings"]["number_of_rom_dofs"] = np.shape(u)[1]
    Dimensions = len(basis_POD["rom_settings"]["nodal_unknowns"])
    N_nodes=np.shape(u)[0]/Dimensions
    N_nodes = int(N_nodes)
    node_Id=np.linspace(1,N_nodes,N_nodes)
    i = 0
    for j in range (0,N_nodes):
        basis_POD["nodal_modes"][int(node_Id[j])] = (u[i:i+Dimensions].tolist())
        i=i+Dimensions

    with open('RomParameters.json', 'w') as f:
        json.dump(basis_POD,f, indent=2)

    print('\n\nNodal basis printed in json format\n\n')















def RunROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    simulation = CFD_ROM(model,parameters)
    #simulation = FluidDynamicsAnalysisROM(model,parameters)
    #print(simulation._GetSolver().settings)
    #print(simulation.project_parameters)
    simulation.Run()
    np.save('SnapshotsMatrixROM.npy', simulation.GetSnapshotsMatrix())
    np.save('control_point_vel_ROM.npy',simulation.GetControlPointVelocityNormHistory())











def TrainHROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    simulation = FluidDynamicsAnalysisROM(model,parameters,"EmpiricalCubature")
    simulation.Run()





def RunHROM():
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = CFD_HROM(model,parameters)
    #simulation = FluidDynamicsAnalysisROM(model,parameters)
    simulation.Run()
    np.save('SnapshotsMatrixHROM.npy', simulation.GetSnapshotsMatrix())
    np.save('control_point_vel_HROM.npy',simulation.GetControlPointVelocityNormHistory())




def RunLWHROM():
    with open("ProjectParametersHROM.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = CFD_HROM(model,parameters)
    #simulation = FluidDynamicsAnalysisROM(model,parameters)
    simulation.Run()
    np.save('SnapshotsMatrixLWHROM.npy', simulation.GetSnapshotsMatrix())
    np.save('control_point_vel_LWHROM.npy',simulation.GetControlPointVelocityNormHistory())



def RunLWHROM2():
    with open("ProjectParametersHROM.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    model = KratosMultiphysics.Model()
    simulation = CFD_HROM(model,parameters)
    #simulation = FluidDynamicsAnalysisROM(model,parameters)
    simulation.Run()
    np.save('SnapshotsMatrixLWHROM2.npy', simulation.GetSnapshotsMatrix())
    np.save('control_point_vel_LWHROM2.npy',simulation.GetControlPointVelocityNormHistory())




def compute_error():
    ROM = np.load('SnapshotsMatrixROM.npy')
    FOM = np.load('SnapshotsMatrix.npy')
    HROM = np.load('SnapshotsMatrixHROM.npy')


    print('approximation error is: ', L2_difference(FOM, ROM))
    print('approximation error is: ', L2_difference(HROM, FOM))





def plot_control_point_velocity_norm2():
    LWHROM = np.load('control_point_vel_LWHROM.npy')
    LWHROM2 = np.load('control_point_vel_LWHROM2.npy')
    diff = LWHROM2 - LWHROM

    # plt.plot(LWHROM,'yo-', label ='LWHROM')
    # plt.plot(LWHROM2,'bo-', label ='LWHROM2')
    plt.plot(diff)
    plt.legend()
    #plt.yscale('log')
    plt.title(r'Velocity Norm at a control point ', fontsize=15, fontweight='bold')
    #plt.title('Singular Values for matrix S1', fontweight='bold')
    plt.xlabel('time step')
    plt.ylabel('velocity norm')
    plt.show()








def plot_control_point_velocity_norm():
    FOM = np.load('control_point_vel_FOM.npy')
    ROM = np.load('control_point_vel_ROM.npy')
    HROM = np.load('control_point_vel_HROM.npy')
    LWHROM = np.load('control_point_vel_LWHROM.npy')

    plt.plot(FOM, 'r', lineWidth=3, label ='FOM')
    plt.plot(ROM,'bo-', label ='ROM')
    plt.plot(HROM,'ko-', label ='HROM')
    plt.plot(LWHROM,'yo-', label ='LWHROM')
    plt.legend()
    #plt.yscale('log')
    plt.title(r'Velocity Norm at a control point ', fontsize=15, fontweight='bold')
    #plt.title('Singular Values for matrix S1', fontweight='bold')
    plt.xlabel('time step')
    plt.ylabel('velocity norm')
    plt.show()



if __name__ == "__main__":
    # start_time = time.time()
    # TrainROM(launch_simulation = True, svd_tolerance = 1e-3)
    # pdb.set_trace()
    # RunROM()
    # TrainHROM()
    # RunHROM()
    # RunLWHROM()
    #RunLWHROM2()
    #end_time = time.time()
    #print(f'It took {end_time - start_time} s to compute the training')
    #compute_error()
    plot_control_point_velocity_norm()
    #plot_control_point_velocity_norm2()


