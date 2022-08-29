import numpy as np
from layers import convolution, pooling, fcl, relu

filter_num = 4
total_step = 0
epochs = 10

params = np.load(f'{filter_num}_{epochs}cnn_params.npz')
test_samples = np.load('hiragana_samples.npz')['arr_1'].reshape((2272,60,60))
labels = np.arange(71)
labels = np.repeat(labels, 32)



conv_len = test_samples.shape[1] - 2
pooling_len = conv_len // 2
fcl_len = (pooling_len**2) * filter_num


conv = convolution.Conv(filter_num)
pooling = pooling.max_pooling()
fcl = fcl.fcl(fcl_len)
relu = relu.relu()

conv.filter = params['arr_0']
fcl.weights = params['arr_1']
fcl.bias = params['arr_2']

accuracy = 0
avg_loss = 0
step = 0

def forward(img):
    output = conv.forward(img)
    output = pooling.forward(relu.forward(output))
    output = fcl.forward(relu.forward(output))

    return output


with open(f'../data/{filter_num}_{epochs}ts.txt','w+') as run_data:
    for img,char in zip(test_samples,labels):
        img = (img - np.min(img)) / np.max(img)-np.min(img) #normalise image between 0 and 1
        output = forward(img)
        loss = -np.log(output[char])
        dlda = np.zeros(71)
        dlda[char] = -1/output[char]

        if np.argmax(output) == char:
            accuracy += 1

        avg_loss += loss

        if step > 0 and step % 32 == 31:
            x = f'{total_step} {avg_loss/32} {(accuracy/32)*100} \n'
            run_data.write(x)
            print(f'Step: {step+1}      Loss: {(avg_loss/32):.5f}     Accuracy: {((accuracy/32)*100):.2f}')
            accuracy = 0
            avg_loss = 0

        step += 1

        