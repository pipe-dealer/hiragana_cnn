import numpy as np
import struct
from PIL import Image, ImageEnhance
import random

def binary_to_img(img_data):
    img = Image.frombytes('F', (128,127), img_data, 'bit', 4) #image has a 4 bit color depth
    img = img.convert('L') #allows it to be converted to .png for debugging
    cont = ImageEnhance.Contrast(img)
    img = cont.enhance(8) #increase contrast in image
    img = img.resize((60,60)) #resize image for faster computation
    return img


#71 hiragana characters, 160 samples per character, each image has a size of 128x127 pixels
hs = np.zeros((71,160,60,60))
shuf_hs = np.zeros((71,160,60,60))
sample = 0


for i in range(1,33): #33 files
    with open(f'ETL8G/ETL8G_{i:02d}','rb') as f:
        if i == 33:
            datasets = 1 #last file only has one dataset
        else:
            datasets = 5 #all other files have 5 datasets
        for dataset in range(datasets):
            char = 0
            for j in range(956): #956 samples per dataset
                b = f.read(8199) #each sample contains 8199 bytes
                c = struct.unpack('>2H8sI4B4H2B30x8128s11x',b)
                if b'.HIRA' in c[2] or b'.WO.' in c[2]: #wo is a hiragana, but did not have the HIRA tag
                    if b'HEI' not in c[2] and b'KAI' not in c[2]: #hei and kai are old hiragana, not used anymore
                        img = binary_to_img(c[14])
                        img = np.array(img)                            
                        hs[char][sample] = img
                        char += 1
            sample += 1

#shuffle entire dataset before splitting to ensure no bias
for i in range(hs.shape[0]):
    index = random.sample(range(hs.shape[1]),hs.shape[1])
    for j in range(hs.shape[1]):
        shuf_hs[i][j] = hs[i][index[j]]



#70% training:testing split
training_samples = shuf_hs[0:71,0:128]
test_samples = shuf_hs[0:71,128:161]

np.savez_compressed('model/hiragana_samples',training_samples,test_samples)

