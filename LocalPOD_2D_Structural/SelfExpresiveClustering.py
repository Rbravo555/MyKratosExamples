import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

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

    if plot:
        init = 0
        for i in range(number_of_trajectories):
            plt.scatter(SnapshotWithTrajectoryInfo[0,init:init+samples_per_trajectory],SnapshotWithTrajectoryInfo[1,init:init+samples_per_trajectory])
            init += samples_per_trajectory
        plt.axis('equal')
        plt.show()

    #fist row of the snapshots is the trajectory it belongs to
    return SnapshotWithTrajectoryInfo


def create_self_expressive_clusters(Snapshots= None, TrajectoryIndex = None, NumberOfTrajectories =  None, number_of_clusters = 12, plot=False):

    if Snapshots is None:
        NumberOfTrajectories =  30
        SamplesPerTrajectory = 10
        S = generating_trajectory_dependent_snapshot_2D(NumberOfTrajectories, SamplesPerTrajectory, plot=False) #last row defines trajectory
        Snapshots = S[:-1,:]
        TrajectoryIndex = S[-1,:]


    Snaps = SelfExpressiveClustering()
    Snaps.GetDataAndTrajectory(Snapshots, TrajectoryIndex, NumberOfTrajectories)
    Snaps.GetDistanceMatrix()
    Snaps.GetKMeansClusters(number_of_clusters)
    Snaps.AddOverlapping()

    if plot:
        for i in range(number_of_clusters):
            plt.title(f'Cluster {i}')
            cluster_i_NO_overlapping = Snaps.Snapshots[:,Snaps.KMeansObject.labels_==i]
            cluster_i_with_overlapping = Snaps.Snapshots[:,Snaps.ClusterIndexesWithOverlapping[i]]
            plt.scatter(cluster_i_with_overlapping[0,:],cluster_i_with_overlapping[1,:], c='red', label = 'overlapping' )
            plt.scatter(cluster_i_NO_overlapping[0,:],cluster_i_NO_overlapping[1,:], c='green', label='original cluster')
            plt.legend()
            plt.xlim([-1,1])
            plt.ylim([-1,1])
            plt.show()
    return Snaps



class SelfExpressiveClustering(object):

    def __init__(self):
        self.Name = 'self expressive clustering + kmeans'

    def GetDataAndTrajectory(self, Snapshots, TrajectoryIndex, NumberOfTrajectories):
        self.Snapshots = Snapshots
        self.TrajectoryIndex = TrajectoryIndex
        self.NumberOfTrajectories = NumberOfTrajectories
        self.LocalNeighbours = []
        self.GlobalNeighbours = []
        for i in range(np.size(self.TrajectoryIndex)):
            self.LocalNeighbours.append([])
            self.GlobalNeighbours.append([])

    def GenerateDataAndTrajectory(self, NumberOfTrajectories=10, samples_per_trajectory=10 ):
        S = generating_trajectory_dependent_snapshot_2D(NumberOfTrajectories, samples_per_trajectory, plot=False) #last row defines trajectory
        self.NumberOfTrajectories = NumberOfTrajectories
        self.Snapshots = S[:-1,:]
        self.TrajectoryIndex = S[-1,:]
        self.LocalNeighbours = []
        self.GlobalNeighbours = []
        for i in range(np.size(self.TrajectoryIndex)):
            self.LocalNeighbours.append([])
            self.GlobalNeighbours.append([])

    def GetSnapshotsForAGivenTrajectory(self, trajectory):
        return self.Snapshots[:,self.GetIndexInAGivenTrajectory(trajectory)]

    def GetIndexInAGivenTrajectory(self,trajectory):
        points_in_trajectory = np.where(self.TrajectoryIndex == trajectory)[0]
        return points_in_trajectory

    def GetSnapshotsNotAGivenTrajectory(self, trajectory):
        return self.Snapshots[:,self.GetIndexNotInAGivenTrajectory(trajectory)]

    def GetIndexNotInAGivenTrajectory(self, trajectory):
        total_number_of_snapshots = np.shape(self.Snapshots)[1]
        full_set_of_indices = np.linspace(0,total_number_of_snapshots-1,total_number_of_snapshots).astype(int)
        return np.setdiff1d(full_set_of_indices, self.GetIndexInAGivenTrajectory(trajectory))

    def GetKMeansClusters(self, number_of_clusters =3):

        self.NumberOfclusters = number_of_clusters
        #TODO run this multiple times and keep the best clustering configuration...

        self.KMeansObject = KMeans(n_clusters=self.NumberOfclusters).fit(self.Snapshots.T)
        #split snapshots into sub-sets
        self.sub_snapshots={}
        for i in range(self.NumberOfclusters):
            self.sub_snapshots[i] = self.Snapshots[:,self.KMeansObject.labels_==i] #we might not neet to save this, can slice it when calling the other function...


        # for i in range(len(self.sub_snapshots)):
        #     plt.scatter(self.sub_snapshots[i][0,:],self.sub_snapshots[i][1,:])

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        # plt.title('Original Clustering', fontsize=15, fontweight ='bold')
        # plt.show()


    def GetDistanceMatrix(self):
        #optimized solution using numpy's broadcasting
        diff = self.Snapshots.reshape(self.Snapshots.shape[0], self.Snapshots.shape[1],1 ) - self.Snapshots.reshape(self.Snapshots.shape[0], 1, self.Snapshots.shape[1])
        self.DistanceMatrix = (diff**2).sum(0)
        for trajectory in range(self.NumberOfTrajectories):
            trajectory_indexes = self.GetIndexInAGivenTrajectory(trajectory)
            other_trajectory_indexes = self.GetIndexNotInAGivenTrajectory(trajectory)
            for j in range(np.size(trajectory_indexes)):
                ordered_idx = np.argsort(self.DistanceMatrix[trajectory_indexes[j],trajectory_indexes])
                #what if there is a tie ?  In case there is a tie, that is, a snapshot is repeated, then any repetition it will be a linear combination and
                #it should be eliminated when doing the SVD afterwards, so it will not matter in the end...
                error = 1
                tolerance = 1e-9
                #calculate the linear independence
                index = 1
                maximum_number_neighbours =11
                snapshot_j = self.Snapshots[:,trajectory_indexes[j]]
                norm_snapshot_j = np.linalg.norm(snapshot_j)
                while error > tolerance:
                    if index==1:
                        local_neighbours = trajectory_indexes[ordered_idx[index]]
                    else:
                        local_neighbours = np.squeeze(np.r_[local_neighbours, trajectory_indexes[ordered_idx[index]]])
                    Q,_ = np.linalg.qr(self.Snapshots[:,local_neighbours].reshape(np.shape(self.Snapshots)[0],-1)) # using numpy's QR for the orthogonal basis
                    if norm_snapshot_j > 0:   #TODO this should be avoided by creating a first cluster containing those snapshots with low norm
                        error = np.linalg.norm(snapshot_j - Q @ Q.T @ snapshot_j) / norm_snapshot_j
                    else:
                        error = np.linalg.norm(snapshot_j - Q @ Q.T @ snapshot_j)

                    index+=1
                    if index>maximum_number_neighbours+1:
                        raise Exception("Too many neighbours :(")


                self.LocalNeighbours[trajectory_indexes[j]] = local_neighbours

            if self.NumberOfTrajectories>1:
                other_trajectory_indexes = self.GetIndexNotInAGivenTrajectory(trajectory)
                for j in range(np.size(trajectory_indexes)):
                    ordered_idx = np.argsort(self.DistanceMatrix[trajectory_indexes[j],other_trajectory_indexes])
                    #what if there is a tie ?  In case there is a tie, that is, a snapshot is repeated, then any repetition will be a linear combination and
                    #it should be eliminated when doing the SVD afterwards, so it will not matter in the end...
                    error = 1
                    tolerance = 1e-9

                    #calculate the linear independence
                    index = 1
                    maximum_number_neighbours = 11
                    snapshot_j = self.Snapshots[:,trajectory_indexes[j]]
                    norm_snapshot_j = np.linalg.norm(snapshot_j)
                    while error > tolerance:
                        if index==1:
                            global_neighbours = other_trajectory_indexes[ordered_idx[index]]
                        else:
                            global_neighbours = np.squeeze(np.r_[global_neighbours, other_trajectory_indexes[ordered_idx[index]]])
                        Q,_ = np.linalg.qr(self.Snapshots[:,global_neighbours].reshape(np.shape(self.Snapshots)[0],-1)) # using numpy's QR for the orthogonal basis
                        if norm_snapshot_j > 0:   #TODO this should be avoided by creating a first cluster containing those snapshots with low norm
                            error = np.linalg.norm(snapshot_j - Q @ Q.T @ snapshot_j) / norm_snapshot_j
                        else:
                            error = np.linalg.norm(snapshot_j - Q @ Q.T @ snapshot_j)

                        index+=1
                        if index>maximum_number_neighbours+1:
                            raise Exception("Too many neighbours :(")
                    self.GlobalNeighbours[trajectory_indexes[j]] = global_neighbours


    def AddOverlapping(self):
        self.OverlapAdded = True
        # join k-means clusters and clusters neighbours
        self.ClusterIndexesWithOverlapping = {}
        for i in range(self.NumberOfclusters):
            OriginalIndexes = np.where(self.KMeansObject.labels_==i)[0]
            for j in range(np.size(OriginalIndexes)): #TODO do this more efficiently
                if j==0:
                    Neighbours = self.LocalNeighbours[OriginalIndexes[j]]
                    if self.NumberOfTrajectories>1:
                        Neighbours = np.squeeze(np.r_[Neighbours, self.GlobalNeighbours[OriginalIndexes[j]]])
                else:
                    Neighbours = np.squeeze(np.r_[Neighbours, self.LocalNeighbours[OriginalIndexes[j]]])
                    if self.NumberOfTrajectories>1:
                        Neighbours = np.squeeze(np.r_[Neighbours, self.GlobalNeighbours[OriginalIndexes[j]]])

            self.ClusterIndexesWithOverlapping[i] = np.unique(np.squeeze(np.r_[OriginalIndexes, Neighbours ]))





if __name__=='__main__':
    snaps = create_self_expressive_clusters()