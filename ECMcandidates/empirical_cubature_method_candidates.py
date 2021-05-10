import numpy as np
import json

try:
    from matplotlib import pyplot as plt
    missing_matplotlib = False
except ImportError as e:
    missing_matplotlib = True

class EmpiricalCubatureMethod():
    """
    This class selects a subset of elements and corresponding positive weights necessary for the construction of a hyper-reduced order model
    Reference: Hernandez 2020. "A multiscale method for periodic structures using domain decomposition and ECM-hyperreduction"
    """


    """
    Constructor setting up the parameters for the Element Selection Strategy
        ECM_tolerance: approximation tolerance for the element selection algorithm
        SVD_tolerance: approximation tolerance for the singular value decomposition of the ResidualSnapshots matrix
        Filter_tolerance: parameter limiting the number of candidate points (elements) to those above this tolerance
        Take_into_account_singular_values: whether to multiply the matrix of singular values by the matrix of left singular vectors. If false, convergence is easier
        Plotting: whether to plot the error evolution of the element selection algorithm

        WOULD ALSO HAVE TO UPDATE THE DOC !!!!!!!!
        SHOULD THE DOC INFO BE PLACED AFTER THE METHOD DECLARATION ? ? ? ?S


    """
    def __init__(self, ECM_tolerance = 1e-6, Filter_tolerance = 1e-16, Plotting = False):
        self.ECM_tolerance = ECM_tolerance
        self.Filter_tolerance = Filter_tolerance
        self.Plotting = Plotting
        self.Name = "EmpiricalCubature"

    """
    Method for setting up the element selection
    input:  ResidualSnapshots: numpy array containing the matrix of residuals projected onto a basis
            OriginalNumberOfElements: number of elements in the original model part. Necessary for the construction of the hyperreduced mdpa
            ModelPartName: name of the original model part. Necessary for the construction of the hyperreduced mdpa
    """
    def SetUp(self, LeftSingularVectorsOfResidualProjected, InitialCandidatesSet = None, OriginalNumberOfElements=None, ModelPartName = None, MaximumMumberUnsuccesfulIterations = 10):
        self.G = LeftSingularVectorsOfResidualProjected
        self.OriginalNumberOfElements = OriginalNumberOfElements
        self.ModelPartName = ModelPartName
        self.y = InitialCandidatesSet
        self.MaximumMumberUnsuccesfulIterations = MaximumMumberUnsuccesfulIterations
        self.b = np.sum(self.G, axis = 1 )
        self.UnsuccesfulIterations = 0


    """
    Method performing calculations required before launching the Calculate method
    """
    def Initialize(self):
        self.Gnorm = np.sqrt(sum(np.multiply(self.G, self.G), 0))
        M = np.shape(self.G)[1]
        normB = np.linalg.norm(self.b)
        GnormNOONE = np.sqrt(sum(np.multiply(self.G[:-1,:], self.G[:-1,:]), 0))
        if self.y is None:
            self.y = np.arange(0,M,1) # Set of candidate points (those whose associated column has low norm are removed)
            if self.Filter_tolerance > 0:
                TOL_REMOVE = self.Filter_tolerance * normB
                rmvpin = np.where(GnormNOONE[self.y] < TOL_REMOVE)
                self.y = np.delete(self.y,rmvpin)
        else:
            self.y_complement = np.arange(0,M,1)
            self.y_complement = np.delete(self.y_complement, self.y)# Set of candidate points (those whose associated column has low norm are removed)
            if self.Filter_tolerance > 0:
                TOL_REMOVE = self.Filter_tolerance * normB
                rmvpin = np.where(GnormNOONE[self.y_complement] < TOL_REMOVE)
                self.y_complement = np.delete(self.y_complement,rmvpin)
        self.z = {}  # Set of intergration points
        self.mPOS = 0 # Number of nonzero weights
        self.r = self.b.copy() # residual vector    This change is necessary, but it was not affecting result (fortunately...)
        self.m = len(self.b) # Default number of points
        self.nerror = np.linalg.norm(self.r)/normB
        self.nerrorACTUAL = self.nerror


    """
    Method launching the element selection algorithm to find a set of elements: self.z, and wiegths: self.w
    """
    def Calculate(self):

        MaximumLengthZ = 0
        k = 1 # number of iterations
        while self.nerrorACTUAL > self.ECM_tolerance and self.mPOS < self.m and len(self.y) != 0:

            #Step 1. Compute new point
            ObjFun = self.G[:,self.y].T @ self.r.T
            ObjFun = ObjFun.T / self.Gnorm[self.y]
            indSORT = np.argmax(ObjFun)
            i = self.y[indSORT]
            if k==1:
                self.w = np.linalg.lstsq(self.G[:, [i]], self.b)[0]
                H = 1/(self.G[:,i] @ self.G[:,i].T)
            else:
                H, self.w = self._UpdateWeightsInverse(self.G[:,self.z],H,self.G[:,i],self.w)


            #Step 3. Move i from set y to set z
            if k == 1:
                self.z = i
            else:
                self.z = np.r_[self.z,i]
            self.y = np.delete(self.y,indSORT)

            # Step 4. Find possible negative weights
            if any(self.w < 0):
                print("WARNING: NEGATIVE weight found")
                indexes_neg_weight = np.where(self.w <= 0.)[0]
                self.y = np.append(self.y, (self.z[indexes_neg_weight]).T)
                self.z = np.delete(self.z, indexes_neg_weight)
                H = self._MultiUpdateInverseHermitian(H, indexes_neg_weight)
                self.w = H @ (self.G[:, self.z].T @ self.b)
                self.w = self.w.reshape(len(self.w),1)

                if  self.UnsuccesfulIterations >  self.MaximumMumberUnsuccesfulIterations:
                    self.y = np.union1d(self.y, self.y_complement) #np.unique(self.y, self.y_complement)
                    print('expanding set to include the complement...')


            if np.size(self.z) > MaximumLengthZ :
                self.UnsuccesfulIterations = 0
            else:
                self.UnsuccesfulIterations += 1

            #Step 6 Update the residual
            if len(self.w)==1:
                self.r = self.b - (self.G[:,self.z] * self.w)
            else:
                Aux = self.G[:,self.z] @ self.w
                self.r = np.squeeze(self.b - Aux.T)
            self.nerror = np.linalg.norm(self.r) / np.linalg.norm(self.b)  # Relative error (using r and b)

            self.nerrorACTUAL = self.nerror  #nerror and nerrprACTUAL are the same, does it make any sense????

            # STEP 7 PLOTTING ONLY
            self.mPOS = np.size(self.z)
            #print(f'k = {k}, m = {np.size(self.z)}, error n(res)/n(b) (%) = {self.nerror*100},  Actual error % = {self.nerrorACTUAL*100} ')

            if k == 1:
                ERROR_GLO = np.array([self.nerrorACTUAL])
                NPOINTS = np.array([np.size(self.z)])
            else:
                ERROR_GLO = np.c_[ ERROR_GLO , self.nerrorACTUAL]
                NPOINTS = np.c_[ NPOINTS , np.size(self.z)]

            MaximumLengthZ = max(MaximumLengthZ, np.size(self.z))
            k = k+1

        #self.w = alpha.T  #SINCE WE USE A VECTOR OF ONES ALWAYS, THIS IS USELES!

        #print(f'Total number of iterations = {k}')

        if missing_matplotlib == False and self.Plotting == True:
            plt.plot(NPOINTS[0], ERROR_GLO[0])
            plt.title('Element Selection Error Evolution')
            plt.xlabel('Number of elements')
            plt.ylabel('Error %')
            plt.show()



    """
    Method for the quick update of weights (self.w), whenever a negative weight is found
    """
    def _UpdateWeightsInverse(self, A,Aast,a,xold):
        c = np.dot(A.T, a)
        d = np.dot(Aast, c).reshape(-1, 1)
        s = np.dot(a.T, a) - np.dot(c.T, d)
        aux1 = np.hstack([Aast + np.outer(d, d) / s, -d / s])
        if np.shape(-d.T / s)[1]==1:
            aux2 = np.squeeze(np.hstack([-d.T / s, 1 / s]))
        else:
            aux2 = np.hstack([np.squeeze(-d.T / s), 1 / s])
        Bast = np.vstack([aux1, aux2])
        v = np.dot(a.T, self.r) / s
        x = np.vstack([(xold - d * v), v])
        return Bast, x



    """
    Method for the quick update of weights (self.w), whenever a negative weight is found
    """
    def _MultiUpdateInverseHermitian(self, invH, neg_indexes):
        neg_indexes = np.sort(neg_indexes)
        for i in range(np.size(neg_indexes)):
            neg_index = neg_indexes[i] - i
            invH = self._UpdateInverseHermitian(invH, neg_index)
        return invH



    """
    Method for the quick update of weights (self.w), whenever a negative weight is found
    """
    def _UpdateInverseHermitian(self, invH, neg_index):
        if neg_index == np.shape(invH)[1]:
            aux = (invH[0:-1, -1] * invH[-1, 0:-1]) / invH(-1, -1)
            invH_new = invH[:-1, :-1] - aux
        else:
            aux1 = np.hstack([invH[:, 0:neg_index], invH[:, neg_index + 1:], invH[:, neg_index].reshape(-1, 1)])
            aux2 = np.vstack([aux1[0:neg_index, :], aux1[neg_index + 1:, :], aux1[neg_index, :]])
            invH_new = aux2[0:-1, 0:-1] - np.outer(aux2[0:-1, -1], aux2[-1, 0:-1]) / aux2[-1, -1]
        return invH_new

