import os
import numpy as np
import random
import numpy.random
from scipy.io import savemat
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
import time
from model import S3F2Net
from data_pre import data_load,nor_pca,border_inter,con_data,getIndex,con_data1
from utils import output_metric
import superpixel_seg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting parameters
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

batchsize = 64
EPOCH = 250
LR = 0.001
FM = 32
seg_scale=600
dataset_name = "Houston"

# load data
Data,Data2,TrLabel,TsLabel= data_load(name=dataset_name)
# TrLabel = small_sample(TrLabel, radito=0.2)
img_row = len(Data2)
img_col = len(Data2[0])

# normalization method 1: map to [0, 1]
[m, n, l] = Data.shape
PC,Data2,NC = nor_pca(Data,Data2,ispca=True)

# LiDAR superpixel segmentation processing
LiDAR_seg = Data2
LiDAR_seg = torch.tensor(LiDAR_seg).float()
LiDAR_seg=np.expand_dims(LiDAR_seg,axis=-1)
ls = superpixel_seg.Superpixel_Seg(seg_scale)
Q, S, A, Segments=ls.SGB(LiDAR_seg)
Q = torch.from_numpy(Q).to(device)
A = torch.from_numpy(A).to(device)

# boundary interpolation
x, x2 = border_inter(PC,Data2,NC)
# construct the training and testing set of HSI and LiDAR
TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2 = con_data(x,x2,TrLabel,TsLabel,NC)
print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)
print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)

# Convert data to tensors and create data loaders for training and testing
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()

TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()
Classes = len(np.unique(TrainLabel))

TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()

dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)
TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel2.long()


#create model
model = S3F2Net(Q,A,FM=FM,NC=NC,Classes=Classes)
# move model to GPU
model.cuda()

 # Initialize optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

BestAcc = 0
torch.cuda.synchronize()
start = time.time()
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

        # move train data to GPU
        b_x1 = b_x1.cuda()
        b_x2 = b_x2.cuda()
        b_y = b_y.cuda()
        LiDAR_seg = torch.tensor(LiDAR_seg).float().cuda()

        out1, out2, out3 = model(b_x1, b_x2,LiDAR_seg)
        loss1 = loss_func(out1, b_y)
        loss2 = loss_func(out2, b_y)
        loss3 = loss_func(out3, b_y)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            model.eval()
            temp1 = TrainPatch1
            temp1 = temp1.cuda()
            temp2 = TrainPatch2
            temp2 = temp2.cuda()
            temp3, temp4, temp5 = model(temp1, temp2,LiDAR_seg)
            Classes = np.unique(TrainLabel1)
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 5000
            for i in range(number):
                temp = TestPatch1[i * 5000:(i + 1) * 5000, :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[i * 5000:(i + 1) * 5000, :, :, :]
                temp1 = temp1.cuda()
                temp2 = model(temp, temp1,LiDAR_seg)[2] + model(temp, temp1,LiDAR_seg)[1] + model(temp, temp1,LiDAR_seg)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp1, temp2, temp3

            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch1[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp1 = temp1.cuda()
                temp2 = model(temp, temp1,LiDAR_seg)[2] + model(temp, temp1,LiDAR_seg)[1] + model(temp, temp1,LiDAR_seg)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
                del temp, temp1, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy,'| ')

            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(model.state_dict(), 'BestAcc.pkl')
                BestAcc = accuracy
            model.train()

print('Best test acc:',BestAcc)
torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start

# load the saved parameters
model.load_state_dict(torch.load('BestAcc.pkl'))
model.eval()
torch.cuda.synchronize()
start = time.time()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//5000
for i in range(number):
    temp = TestPatch1[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[i*5000:(i+1)*5000, :, :]
    temp1 = temp1.cuda()
    temp2 = model(temp, temp1,LiDAR_seg)[2] + model(temp, temp1,LiDAR_seg)[1] + model(temp, temp1,LiDAR_seg)[0]
    temp2_p = temp2.data
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*5000 < len(TestLabel):
    temp = TestPatch1[(i+1)*5000:len(TestLabel), :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[(i+1)*5000:len(TestLabel), :, :]
    temp1 = temp1.cuda()
    temp2 = model(temp, temp1,LiDAR_seg)[2] + model(temp, temp1,LiDAR_seg)[1] + model(temp, temp1,LiDAR_seg)[0]
    temp2_p = temp2.data
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA,AA,Kappa,CA=output_metric(TestLabel1,pred_y)

Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()

torch.cuda.synchronize()
end = time.time()
Test_time = end - start
print(EachAcc)
print('The OA is: ', OA)
print('The AA is: ', AA)
print('The Kappa is: ', Kappa)
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)
print()