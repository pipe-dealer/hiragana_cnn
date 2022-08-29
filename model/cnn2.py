from layers import convolution, pooling, fcl, relu
import numpy as np

data = np.load('hiragana_samples.npz')
params = np.load('6_10cnn_params.npz')

#reshapes the data into a single column of images
training_samples = (data['arr_0'].reshape(9088,60,60)) #10650x60x60
chars = np.arange(71)
chars = np.repeat(chars, 128) #class labels for each image


#test = training_samples[0]

lr = 0.01
filter_num = 6
total_step = 0

#get the input_len for the fcl layer
conv_len = training_samples.shape[1] - 2
pooling_len = conv_len // 2
fcl_len = (pooling_len**2) * filter_num

#initialise layers, with needed parameters
conv = convolution.Conv(filter_num)
pooling = pooling.max_pooling()
fcl = fcl.fcl(fcl_len)
relu = relu.relu()



conv.filter = params['arr_0']
fcl.weights = params['arr_1']
fcl.bias = params['arr_2']

def forward(image):
    output = conv.forward(image)
    output = pooling.forward(relu.forward(output))
    output = fcl.forward(relu.forward(output))
    return output

def backprop(dlda,char,lr):
    dinput = fcl.backprop(dlda,char,lr)
    dinput = pooling.backprop(relu.backprop(dinput))
    dinput = conv.backprop(relu.backprop(dinput),lr)

#create file to store training data
with open('../data/6_10tr.txt','a+') as run_data:
    run_data.write('\n')
    #train for 20 epochs
    for epoch in range(5):
        print(f'EPOCH: {epoch+1}')
        epoch_step = 0
        avg_loss = 0
        accuracy = 0

        #shuffles the training samples and labels the same way so that they still match together
        p = np.random.permutation(9088) #10650 because there are 10650 samples
        training_samples = training_samples[p]
        chars = chars[p]

        for img,char in zip(training_samples,chars):
            img = (img - np.min(img)) / np.max(img)-np.min(img) #normalise image between 0 and 1
            output = forward(img)
            loss = -np.log(output[char])
            dlda = -1/output[char]
            backprop(dlda,char,lr)

            #if index of largest value is equal to label, the network got it correct
            if np.argmax(output) == char:
                accuracy += 1

            avg_loss += loss

            #saves avg acc and loss every 150 steps and prints it out
            if epoch_step > 0 and epoch_step % 128 == 127:
                x = f'{total_step+90879} {avg_loss/128} {(accuracy/128)*100} \n'
                run_data.write(x)
                print(f'Step: {epoch_step+1}      Loss: {(avg_loss/128):.5f}     Accuracy: {((accuracy/128)*100):.2f}')
                accuracy = 0
                avg_loss = 0
            epoch_step += 1
            total_step += 1

#saves updated weights and biases for testing in cnn_params.npz
conv_weights = conv.filter
fcl_weights = fcl.weights
fcl_bias = fcl.bias

np.savez_compressed('6_10cnn_params',conv_weights,fcl_weights,fcl_bias)
run_data.close()