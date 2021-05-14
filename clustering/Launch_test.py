import numpy as np

#importing k-means clustering function from skit-learn
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import random

from scipy.spatial import ConvexHull, convex_hull_plot_2d


def AddOverlapping(SnapshotMatrix, sub_snapshots, kmeans, overlap_percentace=.1):
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


if __name__ == '__main__':
    x,y,g = get_gaussian(random_samples = 20,  plot = False)
    SnapshotsMatrix = build_snapshots_from_3D_data(x,y,g)
    sub_snapshots, kmeans_object = clusterize(SnapshotsMatrix, 3)

    #simple_plot(sub_snapshots)
    #plot_convex_hulls(sub_snapshots, x,y,g)

    sub_snapshots_with_overlapping = AddOverlapping(SnapshotsMatrix, sub_snapshots.copy(), kmeans_object)

    #plot_convex_hulls(sub_snapshots_with_overlapping,x,y,g)
    #simple_plot(sub_snapshots_with_overlapping)

    plot_smallest_cluster_before_and_after_overlapping(sub_snapshots, sub_snapshots_with_overlapping)





