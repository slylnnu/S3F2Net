from scipy import io
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

patchsize1 = 11
patchsize2 = 11
pad_width = np.floor(patchsize1 / 2)
pad_width = np.int32(pad_width)
pad_width2 = np.floor(patchsize2 / 2)
pad_width2 = np.int32(pad_width2)


def data_load(name="Trento"):
    if name == "Houston":
        DataPath1 = './dataset/Houston2013/Houston_HS.mat'
        DataPath2 = './dataset/Houston2013/Houston_LiDAR.mat'
        TRPath = './dataset/Houston2013/Houston_train.mat'
        TSPath = './dataset/Houston2013/Houston_test.mat'

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TR']
        TsLabel = TsLabel['TE']

        Data = io.loadmat(DataPath1)
        Data = Data['input']
        Data = Data.astype(np.float32)

        Data2 = io.loadmat(DataPath2)
        Data2 = Data2['input']
        Data2 = Data2.astype(np.float32)

    elif name == "Trento":
        DataPath1 = './dataset/Trento/HSI.mat'
        DataPath2 = './dataset/Trento/LiDAR.mat'
        TRPath = './dataset/Trento/TRLabel.mat'
        TSPath = './dataset/Trento/TSLabel.mat'

        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

        Data = io.loadmat(DataPath1)
        Data = Data['HSI']
        Data = Data.astype(np.float32)

        Data2 = io.loadmat(DataPath2)
        Data2 = Data2['LiDAR']
        Data2 = Data2.astype(np.float32)

    elif name == "Augsburg":
        DataPath1 = './dataset/Augsburg/data_DSM.mat'
        DataPath2 = './dataset/Augsburg/data_HS_LR.mat'
        TRPath = './dataset/Augsburg/TrainImage.mat'
        TSPath = './dataset/Augsburg/TestImage.mat'
        TrLabel = io.loadmat(TRPath)
        TsLabel = io.loadmat(TSPath)
        TrLabel = TrLabel['TrainImage']
        TsLabel = TsLabel['TestImage']

        Data = io.loadmat(DataPath1)
        Data = Data['data_HS_LR']
        Data = Data.astype(np.float32)

        Data2 = io.loadmat(DataPath2)
        Data2 = Data2['data_DSM']
        Data2 = Data2.astype(np.float32)

    return Data, Data2, TrLabel, TsLabel


def nor_pca(Data, Data2, ispca=True):
    [m, n, l] = Data.shape
    for i in range(l):
        minimal = Data[:, :, i].min()
        maximal = Data[:, :, i].max()
        Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)

    minimal = Data2.min()
    maximal = Data2.max()
    Data2 = (Data2 - minimal) / (maximal - minimal)

    if ispca is True:
        NC = 20
        PC = np.reshape(Data, (m * n, l))
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)
        PC = np.reshape(PC, (m, n, NC))
    else:
        NC = l
        PC = Data

    return PC, Data2, NC


def border_inter(PC, Data2, NC):
    temp = PC[:, :, 0]
    pad_width = np.floor(patchsize1 / 2)
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x = np.empty((m2, n2, NC), dtype='float32')

    for i in range(NC):
        temp = PC[:, :, i]
        pad_width = np.floor(patchsize1 / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x[:, :, i] = temp2

    x2 = Data2
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = np.int32(pad_width2)
    temp2 = np.pad(x2, pad_width2, 'symmetric')
    x2 = temp2
    return x, x2


def  con_data(x, x2, TrLabel, TsLabel, NC):
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        patchlabel = TrLabel[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        patchlabel = TsLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        patchlabel2 = TrLabel[ind1[i], ind2[i]]
        TrainLabel2[i] = patchlabel2

    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        patchlabel2 = TsLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2


def con_data1(x, x2, AllLabel, NC):
    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        Allpatch[i, :, :, :] = patch
        patchlabel = AllLabel[ind1[i], ind2[i]]
        TestLabel[i] = patchlabel

    [ind1, ind2] = np.where(AllLabel != 0)
    TestNum = len(ind1)
    Allpatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum)
    ind3 = ind1 + pad_width2
    ind4 = ind2 + pad_width2
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
        patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        Allpatch2[i, :, :, :] = patch
        patchlabel2 = AllLabel[ind1[i], ind2[i]]
        TestLabel2[i] = patchlabel2

    return Allpatch, Allpatch2


def getIndex(TestLabel, temp):
    index = np.empty(shape=(2, temp), dtype=int)
    k = 0
    for i in range(len(TestLabel)):
        for j in range(len(TestLabel[0])):
            if TestLabel[i][j] != 0:
                index[0][k] = i + 1
                index[1][k] = j + 1
                k += 1

    return index