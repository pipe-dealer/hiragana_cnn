import matplotlib.pyplot as plt

tr_acc = []
tr_losses = []
tr_steps = []

ts_acc = []
ts_losses = []


#change filename to produce results for different runs
with open('data/4_10tr.txt', 'r') as tr, open('data/4_10ts.txt', 'r') as ts:
    for line in tr:
        line = line.split()
        acc = float(line[-1]) / 100
        loss = float(line[1])
        step = float(line[0])
        tr_acc.append(acc)
        tr_losses.append(loss)
        tr_steps.append(step)
    
    for line in ts:
        line = line.split()
        acc = float(line[-1]) / 100
        loss = float(line[1])
        ts_acc.append(acc)
        ts_losses.append(loss)

    tr.close()
    ts.close()

#get average test accuracy and loss
def test_data(ts_acc,ts_losses):
    tavg_acc = sum(ts_acc) * 100 /len(ts_acc)
    tavg_losses = sum(ts_losses)/len(ts_losses)
    return tavg_acc,tavg_losses

#get average training accuracy and loss
def train_data(tr_steps,tr_acc,tr_losses):
    e_acc = []
    e_losses = []
    #averages for each epoch
    for i in range(len(tr_steps)):
        if i % 71 == 70:
            avg_acc = (sum(tr_acc[i-70:i]) * 100) / 71
            avg_loss = sum(tr_losses[i-70:i]) / 71
            e_acc.append(avg_acc)
            e_losses.append(avg_loss)
    
    eavg_acc = e_acc[-1]
    eavg_losses = e_losses[-1]

    return e_acc,e_losses,eavg_acc,eavg_losses

e_acc,e_losses,eavg_acc,eavg_losses = train_data(tr_steps,tr_acc,tr_losses)
tavg_acc,tavg_losses = test_data(ts_acc,ts_losses)
e_steps = range(1,11)

#averages from training and testing
acc_data = f'''
        Training average: {eavg_acc}%\n
        Testing average: {tavg_acc}%
'''

loss_data = f'''
        Training average: {eavg_losses}\n
        Testing average: {tavg_losses}
'''

#create two graphs, for accuracy and loss
_,axs = plt.subplots(2, sharex = True)
plt.title('CNN TRAINING AND TESTING RESULTS')
plt.locator_params(axis='x', nbins=15)

axs[0].plot(e_steps,e_acc)
axs[0].set_title('ACCURACY')
axs[0].set_ylabel('Accuracy (%)')

axs[1].plot(e_steps,e_losses)   
axs[1].set_title('LOSS')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epochs')

axs[0].text(0.54, 0.01, acc_data, transform=axs[0].transAxes, fontsize=7,verticalalignment='bottom') 
axs[1].text(0.55, 0.63, loss_data, transform=axs[1].transAxes, fontsize=7,verticalalignment='bottom')


plt.savefig('data/graphs/final_run.png')

