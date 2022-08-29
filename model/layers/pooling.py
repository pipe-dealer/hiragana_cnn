import numpy as np

class max_pooling:
    #generate image areas
    def pool_iterator(self,img):
        z,h,w = img.shape

        for k in range(z):
            for i in range(h//2): #number of image areas must be equal to total size of feature map
                for j in range(w//2):

                    img_area = img[k][i*2:i*2 + 2, j*2:j*2+2] #2x2

                    yield img_area,k,i,j
        
    
    def forward(self,img):
        self.img = img
        z,h,w = img.shape

        output = np.zeros((z,h//2,w//2)) #8x29x29

        for img_area, k,i,j in self.pool_iterator(img):
            output[k][i][j] = np.max(img_area)
            
        return output
    

    def backprop(self,dlda):
        dldi = np.zeros(self.img.shape) #8x58x58

        for img_area,k,i,j in self.pool_iterator(self.img):
            h,w = img_area.shape
            amax = np.amax(img_area)

            #get position of max value of image area in the input and store that corresponding gradient in the same position in dldi
            for h2 in range(h):
                for w2 in range(w):
                    if img_area[h2][w2] == amax:
                        dldi[k][i*2+h2][j*2+w2] = dlda[k][i][j]

        return dldi