import numpy as np

#importing k-means clustering function from skit-learn
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import random

from scipy.spatial import ConvexHull, convex_hull_plot_2d

import skfuzzy as fuzz





def AddOverlapping(SnapshotMatrix, sub_snapshots, kmeans, overlap_percentace=.2):
    """
    This function could be implemented more efficently
    """
    Number_Of_Clusters = len(sub_snapshots)

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

def ismember(A, B):
    if isinstance(A, np.int_):
        return [ np.sum(A == B) ]
    else:
        return [ np.sum(a == B) for a in A ]


def clusterize(SnapshotMatrix, Number_Of_Clusters = 5):

    kmeans = KMeans(n_clusters=Number_Of_Clusters, random_state=0).fit(SnapshotMatrix.T)
    #split snapshots into sub-sets
    sub_snapshots={}
    for i in range(Number_Of_Clusters):
        sub_snapshots[i] = SnapshotMatrix[:,kmeans.labels_==i] #we might not neet to save this, can slice it when calling the other function...
    return sub_snapshots, kmeans


def gaussians(x,y,sigma = 0.12, mu=0.4):
    g_1 = (1/ sigma**2 *2*np.pi) * np.exp(-( ((x+mu)**2 + (y+mu)**2) / ( 2.0 * sigma**2 ) ) )
    g_2 = (1/ sigma**2 *2*np.pi) * np.exp(-( ((x-mu)**2 + (y-mu)**2) / ( 2.0 * sigma**2 ) ) )
    g_3 = (1/ sigma**2 *2*np.pi) * np.exp(-( ((x+mu)**2 + (y-mu)**2) / ( 2.0 * sigma**2 ) ) )
    g_4 = (1/ sigma**2 *2*np.pi) * np.exp(-( ((x-mu)**2 + (y+mu)**2) / ( 2.0 * sigma**2 ) ) )
    g = g_1 - g_2 -g_3 +  g_4
    return g


def get_gaussian(random_samples = 20, plot = True):
    x = (np.random.rand(random_samples,random_samples) -0.5) *2
    y = (np.random.rand(random_samples,random_samples) -0.5) *2
    g = gaussians(x,y)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_1, y_1 = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
        ax.plot_wireframe(x_1,y_1,gaussians(x_1,y_1), rstride=10, cstride=10)
        ax.scatter(x,y,g, c = g, cmap ='jet')
        # rotate the axes and update
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)

    return x,y,g

def build_snapshots_from_3D_data(x,y,g):
    return np.r_[x.reshape(1,-1), y.reshape(1,-1), g.reshape(1,-1)]


def plot_convex_hulls(sub_snapshots, y, x, g):

    plt.scatter(x,y,c=g, cmap='jet')

    for i in range(len(sub_snapshots)):
        points = sub_snapshots[i][[0,1],:]
        points = points.T

        hull = ConvexHull(points)
        #plt.plot(points[:,0], points[:,1], 'o')
        for simplex in hull.simplices:
            plt.plot(points[simplex, -1], points[simplex, -2], 'k-')

        plt.plot(points[hull.vertices,-1], points[hull.vertices,-2], 'r--', lw=2)
        plt.plot(points[hull.vertices[0],-1], points[hull.vertices[0],-2], 'ro')
    plt.show()

def simple_plot(sub_snapshots):
    for i in range(len(sub_snapshots)):
        plt.scatter(sub_snapshots[i][0,:],sub_snapshots[i][1,:])
    plt.show()

def plot_smallest_cluster_before_and_after_overlapping(sub_snapshots, sub_snapshots_with_overlapping):

    number_of_clusters =len(sub_snapshots)

    for i in range(number_of_clusters):
        plt.scatter(sub_snapshots_with_overlapping[i][0,:],sub_snapshots_with_overlapping[i][1,:], c='red', label = 'overlapping' )
        plt.scatter(sub_snapshots[i][0,:],sub_snapshots[i][1,:], c='green', label='original cluster')
        plt.legend()
        plt.xlim([-1, 1])
        plt.ylim([-1,1])
        plt.show()


def fuzzy_test():
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

    # Define three cluster centers
    centers = [[4, 2],
            [1, 7],
            [5, 6]]

    # Define three cluster sigmas in x and y, respectively
    sigmas = [[0.8, 0.3],
            [0.3, 0.5],
            [1.1, 0.7]]

    # Generate test data
    np.random.seed(42)  # Set seed for reproducibility
    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(200) * i))

    # Visualize the test data
    fig0, ax0 = plt.subplots()
    for label in range(3):
        ax0.plot(xpts[labels == label], ypts[labels == label], '.',
                color=colors[label])
    ax0.set_title('Test data: 200 points x3 clusters.')
    plt.show()


    # Set up the loop and plot
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    alldata = np.vstack((xpts, ypts))
    fpcs = []

    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

        # Store fpc values for later
        fpcs.append(fpc)

        # Plot assigned clusters, for each data point in training set
        cluster_membership = np.argmax(u, axis=0)
        for j in range(ncenters):
            ax.plot(xpts[cluster_membership == j],
                    ypts[cluster_membership == j], '.', color=colors[j])

        # Mark the center of each fuzzy cluster
        for pt in cntr:
            ax.plot(pt[0], pt[1], 'rs')

        ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
        ax.axis('off')

    fig1.tight_layout()
    plt.show()


def generating_trajectory_dependent_snapshot_2D(number_of_trajectories,samples_per_trajectory = 10, plot = True):

    # dummy dataset 1
    SnapshotWithTrajectoryInfo = np.zeros((3,number_of_trajectories*samples_per_trajectory))
    norm_of_vector = 1
    angle_between_trajectories = (2 * np.pi)/number_of_trajectories
    current_angle = 0
    init = 0
    for i in range(number_of_trajectories):
        x_component = norm_of_vector * np.sin(current_angle)
        y_component = norm_of_vector * np.cos(current_angle)
        x = np.linspace(0,x_component,samples_per_trajectory)
        y = np.linspace(0,y_component,samples_per_trajectory)
        SnapshotWithTrajectoryInfo[2,init:init+samples_per_trajectory] = i
        SnapshotWithTrajectoryInfo[0,init:init+samples_per_trajectory] = x
        SnapshotWithTrajectoryInfo[1,init:init+samples_per_trajectory] = y
        init +=samples_per_trajectory
        current_angle += angle_between_trajectories

    if plot == True:
        init = 0
        for i in range(number_of_trajectories):
            plt.scatter(SnapshotWithTrajectoryInfo[0,init:init+samples_per_trajectory],SnapshotWithTrajectoryInfo[1,init:init+samples_per_trajectory])
            init += samples_per_trajectory
        plt.axis('equal')
        plt.show()

    #fist row of the snapshots is the trajectory it belongs to
    return SnapshotWithTrajectoryInfo



def plot_clusterization(Snapshots, k_means_object):
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    for i in range(k_means_object.n_clusters):
        x = Snapshots[0,k_means_object.labels_==i]
        y = Snapshots[1,k_means_object.labels_==i]
        plt.scatter(x,y,c=colors[i])
    plt.show()


# def GetDistance(SnapshotsMatrix):
#     pass


def InterTrajectoryLocalSearch(Snapshots, number_of_trajectories):


    indCANDIDATES_self = []
    indCANDIDATES_rest = []



    for i in range(number_of_trajectories):

        points_in_trajectory_i = np.where(Snapshots[-1,:]==i)[0]
        print(f'points in trajectory {i} : {points_in_trajectory_i} \n')
        number_of_point_in_trajectory_i = np.size(points_in_trajectory_i)
        for j in points_in_trajectory_i: #loop over points in trajectory i
            distance_matrix = Snapshots[:-1,points_in_trajectory_i]  - Snapshots[:-1,j].reshape(Snapshots.shape[0]-1,1)
            distance_from_spap_j_to_snaps_from_traj_i  = np.zeros(number_of_point_in_trajectory_i)
            for k in range(number_of_point_in_trajectory_i):
                distance_from_spap_j_to_snaps_from_traj_i[k] = np.linalg.norm(distance_matrix[:,k])

            ordered_idx = np.argsort(distance_from_spap_j_to_snaps_from_traj_i)
            print(ordered_idx)









    return indCANDIDATES_self, indCANDIDATES_rest




class TrainingSnapshots(object):

    def __init__(self):
        self.Name = 'self expressive clustering'

    def GetDataAndTrajectory(self, Snapshots, trajectories):
        self.Snapshots = Snapshots
        self.Trajectory_index = trajectories

    def GenerateDataAndTrajectory(self):
        S = generating_trajectory_dependent_snapshot_2D(number_of_trajectories=10, samples_per_trajectory=10, plot=False) #last row defines trajectory
        self.Snapshots = S[:-1,:]
        self.Trajectory_index = S[-1,:]

    def GetSnapshotsForAGivenTrajectory(self, trajectory):
        points_in_trajectory = np.where(self.Trajectory_index == trajectory)[0]
        return self.Snapshots[:,points_in_trajectory]

    def GetIndexInAGivenTrajectory(self, trajectory):
        points_in_trajectory = np.where(self.Trajectory_index == trajectory)[0]
        return points_in_trajectory

    def GetIndexNotInAGivenTrajectory(self, trajectory):
        points_in_trajectory = np.where(self.Trajectory_index == trajectory)[0]
        total_number_of_snapshots = np.shape(self.Snapshots)[1]
        full_set_of_indices = np.linspace(0,total_number_of_snapshots-1,total_number_of_snapshots).astype(int)
        return np.setdiff1d(full_set_of_indices, self.GetIndexInAGivenTrajectory(trajectory))

    def CreateClusters(self, number_of_clusters =3):
        _,self.k_means_object = clusterize(self.Snapshots, number_of_clusters)

    def FindSameTrajectoryNeighbours(self, trajectory):

        this_trajectory_indices = self.GetIndexInAGivenTrajectory(trajectory)
        this_trajectory_snapshots = self.GetSnapshotsForAGivenTrajectory(trajectory)
        print(this_trajectory_snapshots)
        plt.scatter(this_trajectory_snapshots[0,:],this_trajectory_snapshots[1,:] )
        plt.show()
        number_of_point_in_this_trajectory = np.size(this_trajectory_indices)


        for j in range(number_of_point_in_this_trajectory): #loop over points in trajectory

            #distance_matrix = this_trajectory_snapshots - this_trajectory_snapshots[:,j].reshape(this_trajectory_snapshots.shape[0],1)
            distance_from_spap_j_to_snaps_from_this_trajectory  = np.zeros(number_of_point_in_this_trajectory)
            for k in range(number_of_point_in_this_trajectory):
                distance_from_spap_j_to_snaps_from_this_trajectory[k] = np.linalg.norm(this_trajectory_snapshots[:,j] - this_trajectory_snapshots[:,k])
            ordered_idx = np.argsort(distance_from_spap_j_to_snaps_from_this_trajectory)
            print(ordered_idx)

        #calculate the linear independence

    def FindOtherTrajectoriesNeighbours(self, trajectory):
        #similarly
        pass









if __name__ == '__main__':

    Snaps = TrainingSnapshots()
    Snaps.GenerateDataAndTrajectory()

    Snaps.CreateClusters()
    #Snaps.FindNeighbours()#same trajectory and other trajectories
    Snaps.FindSameTrajectoryNeighbours(2)
    #Snaps.FindOtherTrajectoriesNeighbours()


    # a = np.array([1,2,3,5,7,8])
    # b = np.array([8,3,6,7,12,6])
    # c = np.r_[a,b,b,a,b]
    # print('c is: ',c,' \nthe unique members in c are',np.unique(c))




    # number_of_trajectories = 10
    # samples_per_trajectory = 20
    # S = generating_trajectory_dependent_snapshot_2D(number_of_trajectories, samples_per_trajectory, plot=False) #last row defines trajectory
    # # S_without_trajectory = S[:2,:] # trim last row (trajectory)
    # # sub_snapshots, kmeans_object = clusterize(S_without_trajectory, 8)
    # # plot_clusterization(S_without_trajectory, kmeans_object)

    # InterTrajectoryLocalSearch( S, number_of_trajectories)







    # x,y,g = get_gaussian(random_samples = 20,  plot = False)
    # SnapshotsMatrix = build_snapshots_from_3D_data(x,y,g)

    # #TODO run algorithm a number of times, and choose the best clustering...  (the one that minimizes some metric)
    # sub_snapshots, kmeans_object = clusterize(SnapshotsMatrix, 3)

    # simple_plot(sub_snapshots)
    # #plot_convex_hulls(sub_snapshots, x,y,g)

    # sub_snapshots_with_overlapping = AddOverlapping(SnapshotsMatrix, sub_snapshots.copy(), kmeans_object)

    # #plot_convex_hulls(sub_snapshots_with_overlapping,x,y,g)
    # #simple_plot(sub_snapshots_with_overlapping)

    # plot_smallest_cluster_before_and_after_overlapping(sub_snapshots, sub_snapshots_with_overlapping)





