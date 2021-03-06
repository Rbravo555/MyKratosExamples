from KratosMultiphysics.RomApplication.element_selection_strategy import ElementSelectionStrategy
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
import KratosMultiphysics

import numpy as np
import json


import pdb

# try:
#     from matplotlib import pyplot as plt
#     missing_matplotlib = False

# except ImportError as e:
#     missing_matplotlib = True

missing_matplotlib = True


class EmpiricalCubatureMethod():

    def __init__(self, ECM_tolerance = 1e-9, Filter_tolerance = 1e-16, Take_into_account_singular_values = False):
        self.ECM_tolerance = ECM_tolerance
        self.Filter_tolerance = Filter_tolerance
        self.Name = "EmpiricalCubature"
        self.RSVDT_Object = RandomizedSingularValueDecomposition()
        self.Take_into_account_singular_values = Take_into_account_singular_values

    def SetUp(self, ResidualSnapshots, weights):
        u , s  = self._ObtainBasis(ResidualSnapshots)

        self.W = weights #np.ones(np.shape(u)[0])
        if self.Take_into_account_singular_values == True:
            G = u[...,:] * np.ones(len(s))
            G = G.T
            G = np.vstack([ G , np.ones( np.shape(G)[1] )]  )
            b = G @ self.W
            bEXACT = b
        else:
            G = u.T
            b = G @ self.W
            bEXACT = b * s
            self.SingularValues = s

        self.b = b
        self.G = G
        self.ExactNorm = np.linalg.norm(bEXACT)

    def Initialize(self):
        self.Gnorm = np.sqrt(sum(np.multiply(self.G, self.G), 0))
        M = np.shape(self.G)[1]
        normB = np.linalg.norm(self.b)
        self.y = np.arange(0,M,1) # Set of candidate points (those whose associated column has low norm are removed)
        GnormNOONE = np.sqrt(sum(np.multiply(self.G[:-1,:], self.G[:-1,:]), 0))
        if self.Filter_tolerance > 0:
            TOL_REMOVE = self.Filter_tolerance * normB
            rmvpin = np.where(GnormNOONE[self.y] < TOL_REMOVE)
            self.y = np.delete(self.y,rmvpin)
        self.z = {}  # Set of intergration points
        self.mPOS = 0 # Number of nonzero weights
        self.r = self.b # residual vector
        self.m = len(self.b) # Default number of points
        self.nerror = np.linalg.norm(self.r)/normB
        self.nerrorACTUAL = self.nerror


    def Calculate(self):
        k = 1 # number of iterations
        while self.nerrorACTUAL > self.ECM_tolerance and self.mPOS < self.m and len(self.y) != 0:

            #Step 1. Compute new point
            ObjFun = self.G[:,self.y].T @ self.r.T
            ObjFun = ObjFun.T / self.Gnorm[self.y]
            indSORT = np.argmax(ObjFun)
            i = self.y[indSORT]
            if k==1:
                alpha = np.linalg.lstsq(self.G[:, [i]], self.b)[0]
                H = 1/(self.G[:,i] @ self.G[:,i].T)
            else:
                H, alpha = self._UpdateWeightsInverse(self.G[:,self.z],H,self.G[:,i],alpha)

            #Step 3. Move i from set y to set z
            if k == 1:
                self.z = i
            else:
                self.z = np.r_[self.z,i]
            self.y = np.delete(self.y,indSORT)

            # Step 4. Find possible negative weights
            if any(alpha < 0):
                print("WARNING: NEGATIVE weight found")
                indexes_neg_weight = np.where(alpha <= 0.)[0]
                self.y = np.append(self.y, (self.z[indexes_neg_weight]).T)
                self.z = np.delete(self.z, indexes_neg_weight)
                H = self._MultiUpdateInverseHermitian(H, indexes_neg_weight)
                alpha = H @ (self.G[:, self.z].T @ self.b)
                alpha = alpha.reshape(len(alpha),1)

            #Step 6 Update the residual
            if len(alpha)==1:
                self.r = self.b - (self.G[:,self.z] * alpha)
            else:
                Aux = self.G[:,self.z] @ alpha
                self.r = np.squeeze(self.b - Aux.T)
            self.nerror = np.linalg.norm(self.r) / np.linalg.norm(self.b)  # Relative error (using r and b)

            if self.Take_into_account_singular_values == False:
                self.nerrorACTUAL = self.SingularValues * self.r
                self.nerrorACTUAL = np.linalg.norm(self.nerrorACTUAL / self.ExactNorm )


            self.nerrorACTUAL = self.nerror

            # STEP 7
            self.mPOS = np.size(self.z)
            print(f'k = {k}, m = {np.size(self.z)}, error n(res)/n(b) (%) = {self.nerror*100},  Actual error % = {self.nerrorACTUAL*100} ')

            if k == 1:
                ERROR_GLO = np.array([self.nerrorACTUAL])
                NPOINTS = np.array([np.size(self.z)])
            else:
                ERROR_GLO = np.c_[ ERROR_GLO , self.nerrorACTUAL]
                NPOINTS = np.c_[ NPOINTS , np.size(self.z)]

            k = k+1

        self.w = alpha.T #* np.sqrt(self.W[self.z])


    def _UpdateWeightsInverse(self, A, Aast,a,xold):
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

    def _MultiUpdateInverseHermitian(self, invH, neg_indexes):
        neg_indexes = np.sort(neg_indexes)
        for i in range(np.size(neg_indexes)):
            neg_index = neg_indexes[i] - i
            invH = self._UpdateInverseHermitian(invH, neg_index)
        return invH

    def _UpdateInverseHermitian(self, invH, neg_index):
        if neg_index == np.shape(invH)[1]:
            aux = (invH[0:-1, -1] * invH[-1, 0:-1]) / invH(-1, -1)
            invH_new = invH[:-1, :-1] - aux
        else:
            aux1 = np.hstack([invH[:, 0:neg_index], invH[:, neg_index + 1:], invH[:, neg_index].reshape(-1, 1)])
            aux2 = np.vstack([aux1[0:neg_index, :], aux1[neg_index + 1:, :], aux1[neg_index, :]])
            invH_new = aux2[0:-1, 0:-1] - np.outer(aux2[0:-1, -1], aux2[-1, 0:-1]) / aux2[-1, -1]
        return invH_new

    def _ObtainBasis(self,ResidualSnapshots):
        u,s,_,_ = self.RSVDT_Object.Calculate(ResidualSnapshots, 1e-9)
        return u, s


    def WriteSelectedElements(self, w, z, OriginalNumberOfElements):
        pdb.set_trace()
        w = np.squeeze(w)
        ### Saving Elements and conditions
        ElementsAndWeights = {}
        ElementsAndWeights["Elements"] = {}
        ElementsAndWeights["Conditions"] = {}

        for j in range (0,len(z)):
            if z[j] <= OriginalNumberOfElements-1:
                ElementsAndWeights["Elements"][int(z[j])] = (float(w[j]))
            else:
                ElementsAndWeights["Conditions"][int(z[j])-OriginalNumberOfElements] = (float(w[j]))

        with open('ElementsAndWeights.json', 'w') as f:
            json.dump(ElementsAndWeights,f, indent=2)
        print('\n\n Elements and conditions selected have been saved in a json file\n\n')
        #self._CreateHyperReducedModelPart()

    #Good for fluid
    def _CreateHyperReducedModelPart(self, ModelPartName):
        self.ModelPartName = ModelPartName
        current_model = KratosMultiphysics.Model()
        computing_model_part = current_model.CreateModelPart("main")
        model_part_io = KratosMultiphysics.ModelPartIO(self.ModelPartName)
        model_part_io.ReadModelPart(computing_model_part)
        hyper_reduced_model_part_help =   current_model.CreateModelPart("Helping")



        with open('ElementsAndWeights.json') as f:
            HR_data = json.load(f)
            for key in HR_data["Elements"].keys():
                for node in computing_model_part.GetElement(int(key)+1).GetNodes():
                    hyper_reduced_model_part_help.AddNode(node,0)
            for condition in computing_model_part.Conditions:
                for node in condition.GetNodes():
                    hyper_reduced_model_part_help.AddNode(node, 0)
                #hyper_reduced_model_part_help.AddCondition(condition, 0)

        # The HROM model part. It will include two sub-model parts. One for caculation, another one for visualization
        HROM_Model_Part =  current_model.CreateModelPart("HROM_Model_Part")

        # Building the COMPUTE_HROM submodel part
        hyper_reduced_model_part = HROM_Model_Part.CreateSubModelPart("COMPUTE_HROM")

        with open('ElementsAndWeights.json') as f:
            HR_data = json.load(f)
            for originalSubmodelpart in computing_model_part.SubModelParts:
                hyperReducedSubmodelpart = hyper_reduced_model_part.CreateSubModelPart(originalSubmodelpart.Name)
                print(f'originalSubmodelpart.Name {originalSubmodelpart.Name}')
                print(f'originalSubmodelpart.Elements {len(originalSubmodelpart.Elements)}')
                print(f'originalSubmodelpart.Conditions {len(originalSubmodelpart.Conditions)}')
                for originalNode in originalSubmodelpart.Nodes:
                    if originalNode in hyper_reduced_model_part_help.Nodes:
                        hyperReducedSubmodelpart.AddNode(originalNode,0)
                ## More eficient way to implement this is possible
                for originalElement in originalSubmodelpart.Elements:
                    for key in HR_data["Elements"].keys():
                        if originalElement.Id == int(key)+1:
                            hyperReducedSubmodelpart.AddElement(originalElement,0)
                            print(f'For the submodelpart {hyperReducedSubmodelpart.Name}, the element with the Id {originalElement.Id} is assigned the key {key}')
                for originalCondition in originalSubmodelpart.Conditions:
                    hyperReducedSubmodelpart.AddCondition(originalCondition,0)
                    print(f'For the submodelpart {hyperReducedSubmodelpart.Name}, the condition with the Id {originalCondition.Id} is assigned the key {key}')


        # # Building the VISUALIZE_HROM submodel part
        # print('Adding skin for visualization...')
        # hyper_reduced_model_part2 = HROM_Model_Part.CreateSubModelPart("VISUALIZE_HROM")
        # for condition in computing_model_part.Conditions:
        #     for node in condition.GetNodes():
        #         hyper_reduced_model_part2.AddNode(node, 0)
        #     hyper_reduced_model_part2.AddCondition(condition, 0)
        # # for node in computing_model_part.Nodes:
        # #     hyper_reduced_model_part2.AddNode(node, 0)

        ## Creating the mdpa file using ModelPartIO object
        print('About to print ...')
        KratosMultiphysics.ModelPartIO("Hyper_Reduced_Model_Part", KratosMultiphysics.IO.WRITE| KratosMultiphysics.IO.MESH_ONLY ).WriteModelPart(HROM_Model_Part)
        print('\nHyper_Reduced_Model_Part.mdpa created!\n')
        KratosMultiphysics.kratos_utilities.DeleteFileIfExisting("Hyper_Reduced_Model_Part.time")


if __name__=='__main__':

    ECM_object = EmpiricalCubatureMethod()

