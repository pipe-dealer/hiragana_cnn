import numpy as np
import math

class Conv:
    def __init__(self,filter_num):
        self.filter_num = filter_num

        #kaiming initialization for filter weights
        self.filter = np.random.normal(0,math.sqrt(2/3),size=(filter_num,3,3)) #8x3x3

    #gets all image areas to be applied to the filter
    def filter_mapping(self,img):
        h,w = img.shape

        for i in range(h-2): #filter will not be able to cover entire image
            for j in range(w-2):
                img_area = img[i:i+3, j:j+3]
                yield img_area, i,j 

    def forward(self,img):
        self.img = img
        h,w = img.shape #get dimensions of image

        #create empty array for feature map
        output = np.zeros((self.filter_num,h-2,w-2,)) #8x58x58

        filter_map = self.filter_mapping(img)
        for k in range(self.filter_num):
            for img_area,i,j in filter_map:
                #convolution operation
                output[k][i][j] = np.tensordot(img_area,self.filter[k])
        return output
    
    def backprop(self,dlda,lr):
        dldf = np.zeros(self.filter.shape) #8x3x3

        for k in range(self.filter_num):
            #convolution operation, except between image and output gradients
            for img_area,i,j in self.filter_mapping(self.img):
                dldf[k] = np.sum(dlda[k][i][j] * img_area)



        self.filter -= lr * dldf

