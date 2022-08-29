import numpy as np

class relu:
    #leaky relu
    def relu(self,x):
        return np.where(x>0,x,x*0.01) #if value is greater than zero, keep it the same, else multiiply it by 0.01
    #leaky relu derivative
    def drelu(self,x):
        return np.where(x>0,1,0.01) #if value is greater than zero, set it to one, else change it to 0.01
    
    def forward(self,img):
        return self.relu(img)
    
    def backprop(self,dlda):
        return dlda * self.drelu(dlda)