from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import KratosMultiphysics
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from scipy import linalg
from matplotlib import pyplot as plt
import numpy as np
import KratosMultiphysics.RomApplication as romapp
import json
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
import pdb
import time

#importing k-means clustering function from skit-learn
from sklearn.cluster import KMeans


class StructuralMechanicsAnalysisSavingData(StructuralMechanicsAnalysis):

    def __init__(self,model,project_parameters):
        super(StructuralMechanicsAnalysisSavingData,self).__init__(model,project_parameters)
        self.time_step_solution_container = []

    def FinalizeSolutionStep(self):
        super(StructuralMechanicsAnalysisSavingData,self).FinalizeSolutionStep()
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


def ismember(A, B):
    if isinstance(A, np.int_):
        return [ np.sum(A == B) ]
    else:
        return [ np.sum(a == B) for a in A ]

def AddOverlapping(SnapshotMatrix, sub_snapshots, Number_Of_Clusters, kmeans, overlap_percentace=.2):
    """
    This function could be implemented more efficently
    """
    if Number_Of_Clusters!=1:

        neighbors={}
        for i in range(Number_Of_Clusters):
            neighbors[i] = []

        #Add overlapping
        for i in range(np.shape(SnapshotMatrix)[1]):
            #identify the two nearest cluster centroids to state i and mark these clusters as neighbors
            this_matrix = (kmeans.cluster_centers_).T - SnapshotMatrix[:,i].reshape(np.shape(SnapshotMatrix)[0], 1)
            distance = np.zeros((Number_Of_Clusters))
            for j in range(Number_Of_Clusters):
                distance[j] = np.linalg.norm(this_matrix[:,j])
            second_nearest_idx = np.argsort(distance)[1]
            if not(sum(ismember(neighbors[kmeans.labels_[i]],second_nearest_idx))):
                neighbors[kmeans.labels_[i]].append(second_nearest_idx)

        snapshots_to_add = []
        for j in range(Number_Of_Clusters):
            N_snaps = np.shape(sub_snapshots[j])[1]
            N_neighbors = len(neighbors[j])

            N_add = int(np.ceil(N_snaps*overlap_percentace / N_neighbors))#number of snapshots to add to subset i

            for i in range(N_neighbors):
                print('adding neighbors from ', neighbors[j][i], 'to cluster ', j  )
                this_matrix = sub_snapshots[neighbors[j][i]] - ((kmeans.cluster_centers_[j]).T).reshape(np.shape(SnapshotMatrix)[0], 1)
                distance = np.zeros(np.shape(sub_snapshots[neighbors[j][i]][1]))
                for k in range(len(distance)):
                    distance[k] = np.linalg.norm(this_matrix[:,k])
                indices_to_add = np.argsort(distance)
                if i==0:
                    snapshots_to_add.append(sub_snapshots[neighbors[j][i]][:,indices_to_add[:N_add]])
                else:
                    snapshots_to_add[j] =  np.c_[ snapshots_to_add[j] , sub_snapshots[neighbors[j][i]][:,indices_to_add[:N_add]] ]

        for j in range(Number_Of_Clusters):
            sub_snapshots[j] = np.c_[sub_snapshots[j], snapshots_to_add[j]]

    return sub_snapshots

def Pre_compute_distances(u0, kmeans, bases):

    number_of_bases = len(kmeans.cluster_centers_)
    d = {}
    w = {}
    for i in range(number_of_bases):
        w[i]={}
        d[i]={}
        for j in range(number_of_bases):
            w[i][j]={}
            d[i][j]=-1
            for m in range(number_of_bases):
                w[i][j][m]=[0]

    regularized_centers = kmeans.cluster_centers_ - u0.T
    for m in range(number_of_bases):
        k = m+1
        for p in range(k,number_of_bases):
            d[m][p] =  (regularized_centers[m].T @ regularized_centers[m]) - (regularized_centers[p].T @ regularized_centers[p])
            d[p][m] = -1*d[m][p]
            for l in range(number_of_bases):
                w[l][m][p] = (2 * bases[l].T @ (regularized_centers[p].T - regularized_centers[m] )).tolist()
    return w , d

def set_up_q_kratos_format(q):
    number_of_entries = len(q)
    q_kratos = KratosMultiphysics.Vector(number_of_entries)
    for i in range(number_of_entries):
        q_kratos[i] = q[i]
    return q_kratos


def return_dof_in_cpp_using_node_id(node_id, number_of_dofs =2):
    node_id = node_id -1
    node_id = list(range(node_id*number_of_dofs,(node_id*number_of_dofs)+number_of_dofs))
    return node_id


def convert_to_2d(SnapshotsMatrix, NumberOfDimensions=2):
    for i in range(np.shape(SnapshotsMatrix)[1]):
        column_mean = np.mean( SnapshotsMatrix[:,i].reshape(-1,NumberOfDimensions).reshape(-1,2),0).reshape(-1,1)
        if i ==0:
            columns_means = column_mean
        else:
            columns_means = np.c_[columns_means,column_mean]

    return columns_means



if __name__ == "__main__":
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    model = KratosMultiphysics.Model()
    simulation = StructuralMechanicsAnalysisSavingData(model,parameters)
    start = time.time()
    simulation.Run()
    end = time.time()
    print(end - start)
    pdb.set_trace()
    SnapshotMatrix = simulation.EvaluateQuantityOfInterest()


    np.save('SnapshotMatrix.npy', SnapshotMatrix)
    #SnapshotMatrix = np.load('SnapshotMatrix.npy')
    norm_all = np.linalg.norm(SnapshotMatrix, 'fro')


    add_overlapping = False

    # #clustering
    Number_Of_Clusters = 3
    kmeans = KMeans(n_clusters=Number_Of_Clusters, random_state=0).fit(SnapshotMatrix.T)

    #split snapshots into sub-sets
    sub_snapshots={}
    for i in range(Number_Of_Clusters):
        sub_snapshots[i] = SnapshotMatrix[:,kmeans.labels_==i] #we might not neet to save this, can slice it when calling the other function...

    #add overlapping 10%
    if add_overlapping:
        sub_snapshots = AddOverlapping(SnapshotMatrix, sub_snapshots,Number_Of_Clusters, kmeans)

    #calcualte the svd of each cluster and obtain its modes
    Bases={}
    tolerances = []

    for i in range(Number_Of_Clusters):
        norm_cluster = np.linalg.norm(sub_snapshots[i], 'fro')
        tolerance_i = (norm_all/norm_cluster)
        tolerances.append(tolerance_i)
        print(tolerance_i)
        #Bases[i],s,_,_ = RandomizedSingularValueDecomposition().Calculate(sub_snapshots[i], 1e-4 * tolerance_i     )
        #Bases[i],s,_,_ = RandomizedSingularValueDecomposition().Calculate(sub_snapshots[i], 1e-6  )
        Bases[i],s,_= np.linalg.svd(sub_snapshots[i], full_matrices=False)
        Bases[i] = Bases[i][:,:8]
        s = s[:8]
        np.save(f'Bases_{i+1}.npy', Bases[i])
        #Bases[i] = np.load(f'Bases_{i+1}.npy')
        # print(Bases[i])



    #checking how well are basis being approximated
    for i in range(Number_Of_Clusters):
        print('The error of representation basis ',i, 'using the basis ', 1, ' is ',   np.linalg.norm(      Bases[i] -  Bases[0]@Bases[0].T@Bases[i])  / np.linalg.norm(Bases[i]))


    pdb.set_trace()

    elements_to_print = [1]
    nodes_to_print = [2,4,1]



    dofs_to_print = []
    for i in range(len(nodes_to_print)):
        dofs_to_print.append(return_dof_in_cpp_using_node_id(nodes_to_print[i]))
    dofs_to_print = [item for sublist in dofs_to_print for item in sublist]




    for k in range(len(elements_to_print)):
        for i in range(len(Bases)):
            print('\n', i, 'th Nodal Basis for elemen ', elements_to_print[k])
            for j in range(len(nodes_to_print)):
                print('node ',nodes_to_print[j] )
                print(Bases[i][return_dof_in_cpp_using_node_id(nodes_to_print[j]),:])


    #visualize clusterization in 2D
    two_d_snapthots = convert_to_2d(SnapshotMatrix)
    plt.scatter(two_d_snapthots[0,:], two_d_snapthots[1,:], c=kmeans.labels_, s=50, cmap='viridis')
    centroids_to_plot = convert_to_2d((kmeans.cluster_centers_).T)
    plt.scatter(centroids_to_plot[0,:], centroids_to_plot[1,:], c='black', s=200, alpha=0.5)
    plt.title('clustering visualization')
    plt.show()

    np.save('labels.npy', kmeans.labels_)
    np.save('centroids_to_plot.npy', centroids_to_plot)
    np.save('two_d_snapthots.npy', two_d_snapthots)




    ### Saving the nodal basis ###  (Need to make this more robust, hard coded here)
    basis_POD={"rom_settings":{},"nodal_modes":{}}
    basis_POD["rom_settings"]["nodal_unknowns"] = ["DISPLACEMENT_X","DISPLACEMENT_Y"]
    basis_POD["rom_settings"]["number_of_rom_dofs"] = []
    for i in range(Number_Of_Clusters):
        basis_POD["rom_settings"]["number_of_rom_dofs"].append(np.shape(Bases[i])[1])

    Dimensions = len(basis_POD["rom_settings"]["nodal_unknowns"])
    N_nodes=np.shape(Bases[0])[0]/Dimensions
    N_nodes = int(N_nodes)
    node_Id=np.linspace(1,N_nodes,N_nodes)
    i = []
    for l in range(Number_Of_Clusters):
        i.append(0)

    basis_POD["nodal_modes"]={}
    for j in range (0,N_nodes):
        basis_POD["nodal_modes"][int(node_Id[j])] = {}
        for k in range(Number_Of_Clusters):
            basis_POD["nodal_modes"][int(node_Id[j])][k] = (Bases[k][i[k]:i[k]+Dimensions].tolist())
            print(i)
            i[k]+=Dimensions
            print(i)

    w,z0 = Pre_compute_distances(SnapshotMatrix[:,0], kmeans, Bases)
    print(z0)

    basis_POD["w"] = w
    basis_POD["z0"] = z0

    basis_POD["cluster_centroids"] = kmeans.cluster_centers_.tolist()

    with open('RomParameters.json', 'w') as f:
        json.dump(basis_POD,f, indent=2)

    print('\n\nNodal basis printed in json format\n\n')



    #get the Delta_s's
    Delta_u = np.zeros(np.shape(SnapshotMatrix))
    for i in range(np.shape(SnapshotMatrix)[1]):
        if i>0:
            Delta_u[:,i] = SnapshotMatrix[:,i] - SnapshotMatrix[:,i-1]
        else:
            Delta_u[:,i] = SnapshotMatrix[:,i]

    #load the pre-computed distance operators
    with open('RomParameters.json') as f:
        data = json.load(f)

        number_of_bases = Number_Of_Clusters
        distance_to_clusters = romapp.DistanceToClusters(number_of_bases)
        w = data["w"]
        z0 = data["z0"]
        for i in range(number_of_bases):
            for j in range(number_of_bases):
                distance_to_clusters.SetZEntry(z0[f'{i}'][f'{j}'],i,j)
                for k in range(number_of_bases):
                    entries = len(w[f'{i}'][f'{j}'][f'{k}'])
                    temp_vector = KratosMultiphysics.Vector(entries)
                    for entry in range(entries):
                        temp_vector[entry] = w[f'{i}'][f'{j}'][f'{k}'][entry]
                    distance_to_clusters.SetWEntry(temp_vector,i,j,k)
        print(distance_to_clusters.GetZMatrix())



    pdb.set_trace()

    current_cluster = []
    Delta_q = []
    Z_matrix = []
    #Use C++ opeartions to update the current cluster
    for i in range(np.shape(Delta_u)[1]):
        Z_matrix.append(np.array(distance_to_clusters.GetZMatrix(), copy = True))
        distance_to_clusters.UpdateCurrentCluster()
        current_cluster.append(distance_to_clusters.GetCurrentCluster())
        Delta_q.append((Bases[current_cluster[i]].T @ Delta_u[:,i]).tolist())
        q_kratos = set_up_q_kratos_format(Delta_q[i])
        distance_to_clusters.UpdateZMatrix(q_kratos)

    for i in range(len(kmeans.labels_)):
        print('Snapshot ',i,' is in cluster ', current_cluster[i])
        print('The delta q should be: ', Delta_q[i])
        print('Z matrix is: ', Z_matrix[i])
        print('Snapshot ',i,' should be in cluster ',kmeans.labels_[i],'\n')


    basis_POD["delta_q"] = Delta_q
    basis_POD["correct_cluster"] = current_cluster
    basis_POD["nodes_to_print"] = nodes_to_print
    basis_POD["elements_to_print"] = elements_to_print

    with open('RomParameters.json', 'w') as f:
        json.dump(basis_POD,f, indent=2)


    # for i in range(np.shape(SnapshotMatrix)[1]):
    # if i>0:
    #     Delta_u = Bases[kmeans.labels_[i]]@ (Bases[kmeans.labels_[i]].T @ SnapshotMatrix[:,i]) - (Bases[kmeans.labels_[i]] @ (Bases[kmeans.labels_[i]].T @ SnapshotMatrix[:,i-1]))
    # else:
    #     Delta_u = Bases[kmeans.labels_[i]]@ (Bases[kmeans.labels_[i]].T @ SnapshotMatrix[:,i])

    # Delta_q =  (Bases[kmeans.labels_[i]]).T @ Delta_u
