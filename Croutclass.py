import numpy as np

class Crout():

    #Get the augmented matrix
    def AM(self, A, b):
        self.size = len(b)
        for i in range(0, self.size):
            A[i].append(b[i])
        self.matrix = np.array(A, dtype = "float64")
        return self.matrix
    
    def LUz(self):
        L = [[0 for i in range(0, self.size)] for j in range(0,self.size)]
        self.U = np.identity(self.size)
        self.z = [0 for i in range(0,self.size)] 

        L[0][0] = self.matrix[0][0]
        self.U[0][1] = self.matrix[0][1]/L[0][0]
        self.z[0] = self.matrix[0][self.size]/L[0][0]

        for i in range(1,self.size-1):
            L[i][i-1] = self.matrix[i][i-1]
            L[i][i] = self.matrix[i][i] - L[i][i-1]*self.U[i-1][i]
            self.U[i][i+1] = self.matrix[i][i+1]/L[i][i]
            self.z[i] = (self.matrix[i][self.size] - L[i][i-1]*self.z[i-1])/L[i][i]

        L[self.size-1][self.size-2] = self.matrix[self.size-1][self.size-2]
        L[self.size-1][self.size-1] = self.matrix[self.size-1][self.size-1] - L[self.size-1][self.size-2]*self.U[self.size-2][self.size-1]
        self.z[self.size-1] = (self.matrix[self.size-1][self.size] - L[self.size-1][self.size-2]*self.z[self.size-2])/L[self.size-1][self.size-1]

        return L, self.U, self.z

    def x_vals(self):
        x = [0 for i in range(0,self.size)]
        x[-1] = self.z[-1]
        for i in reversed(range(0, self.size-1)):
            x[i] = self.z[i] - self.U[i][i+1]*x[i+1]

        return x
