import numpy as np
import math

class fcl:
    def __init__(self,input_len):
        self.input_len = input_len

        # xavier initialization
        self.weights = np.random.uniform(-1/math.sqrt(input_len),1/math.sqrt(input_len),size=(71,input_len)) #71x3364


        self.bias = np.zeros(71)
    
    #softmax
    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x))

    #derivitve of softmax
    def dsoftmax(self,z,char):
        sigma = np.sum(np.exp(z))
        dadz = np.exp(z) * -np.exp(z[char]) / sigma**2

        #correct class derivative is calculated differently
        dadz[char] = np.exp(z[char]) * (sigma - np.exp(z[char])) / sigma ** 2
        return dadz
        
    def forward(self,img):
        self.img_shape = img.shape #8x29x29
        img = img.flatten()
        self.img = img
        #weight x input + bias
        self.z = np.dot(self.weights,img) + self.bias
        a = self.softmax(self.z)

        return a
    
    def backprop(self,dlda,char,lr):
        #d means partial derivitve

        dadz = self.dsoftmax(self.z,char) #1x71

        dzdw = self.img #1x3364
        dzdb = 1
        dzdx = self.weights #71x3364

        dldz = dlda * dadz #1x71

        #newaxis and .T changes the shape of the matrices so that they meet the conditions for matrix multiplication
        dldw = np.dot(dldz[np.newaxis].T, dzdw[np.newaxis]) #71x3364
        dldb = dldz * dzdb
        dldx = np.dot(dldz.T, dzdx)

        self.weights -= lr * dldw
        self.bias -= lr * dldb

        dldx = dldx.reshape(self.img_shape) #8x29x29
        return dldx