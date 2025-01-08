import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class GCN_1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCN_1, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # mask=torch.ceil(A * 0.00001)
        # nodes_count =A.shape[0]
        # I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # A = torch.where(mask > 0, e, zero_vec) + I
        # if model != 'normal': A = torch.clamp(A, 0.1)  # This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A

class GCN_2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,A:torch.Tensor):
        super(GCN_2, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # 方案一：minmax归一化
        H = self.BN(H)
        H_xx1= self.GCN_liner_theta_1(H)
        A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        # H = self.BN(H)
        # H_xx1 = self.GCN_liner_theta_1(H)
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # mask=torch.ceil(A * 0.00001)
        # nodes_count =A.shape[0]
        # I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        # A = torch.where(mask > 0, e, zero_vec) + I
        # if model != 'normal': A = torch.clamp(A, 0.1)  # This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        # output = self.Activition(torch.mm(A, self.GCN_liner_out_1(H)))

        return output, A
class S3F2Net(nn.Module):
    def __init__(self,Q,A,FM,NC,Classes):
        super(S3F2Net, self).__init__()
        self.Q=Q
        self.A=A
        self.Classes=Classes
        input_dim=8
        output_dim=8
        dim=8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = NC,out_channels = FM,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 1, 1, ),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN', nn.BatchNorm2d(1))
        self.CNN_denoise.add_module('CNN_denoise_Conv', nn.Conv2d(1, dim, kernel_size=(3, 3), stride=1, padding=1))
        self.CNN_denoise.add_module('CNN_denoise_Act', nn.LeakyReLU())

        self.out1 = nn.Linear(FM*4,Classes)
        self.out2 = nn.Linear(FM*4,Classes)
        self.out3 = nn.Linear(FM*4,Classes)

        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
        self.linear_1 = nn.Linear(dim, 1)
        self.linear_2 = nn.Linear(self.A.size(0), 1)
        self.linear_3 = nn.Linear(dim, self.Classes)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.GCN_Branch_1 = GCN_1(input_dim, output_dim,self.A)
        self.GCN_Branch_2 = GCN_2(input_dim, output_dim,self.A)

    def forward(self, x1, x2, x3):
        #Convolutional neural network processing procedure
        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        xh = self.conv3(x1)
        xl = self.conv6(x2)

        #Shared weight convolution
        x1 = self.conv7(x1)
        x2 = self.conv7(x2)
        x_fusion = x1 + x2

        # Local Feature Prediction
        xh = xh.view(xh.size(0), -1)
        CNN_result_1 = self.out1(xh)

        xl = xl.view(x2.size(0), -1)
        CNN_result_2 = self.out2(xl)

        x_fusion = x_fusion.view(x_fusion.size(0), -1)
        CNN_result_3 = self.out3(x_fusion)

        # Graph convolutional network processing procedure
        # Denoising the input
        denoised_input = self.CNN_denoise(torch.unsqueeze(x3.permute([2, 0, 1]), 0))
        denoised_input = torch.squeeze(denoised_input, 0).permute([1, 2, 0])

        # Flattening the denoised input for further processing
        flattened_input = denoised_input.view(denoised_input.size(0) * denoised_input.size(1), denoised_input.size(2))

        # Computing the initial features for GCN using the normalized association matrix
        initial_features = torch.mm(self.norm_col_Q.t(), flattened_input)

        # Passing through the first GCN branch
        H1, _ = self.GCN_Branch_1(initial_features)
        scores = self.linear_1(H1)

        # Sorting the scores and applying threshold normalization
        sorted_scores, sorted_indices = torch.sort(scores, dim=0)
        threshold = self.linear_2(sorted_scores.permute([1, 0]))
        sorted_scores = sorted_scores - threshold

        # Reversing the sorting indices to reconstruct original order
        _, inverse_sorted_indices = torch.sort(sorted_indices, dim=0)
        weighted_features = sorted_scores * H1[sorted_indices].squeeze()
        recovered_features = weighted_features[inverse_sorted_indices].squeeze()

        # Calculating the cosine similarity matrix
        similarity_matrix = F.cosine_similarity(H1.unsqueeze(1), H1.unsqueeze(0), dim=-1)

        # Combining the similarity matrix with the reordered features to update the features
        Integrated_features = torch.matmul(similarity_matrix, recovered_features)
        updated_features = Integrated_features + recovered_features

        # Passing the updated features through the second GCN branch
        H2, _ = self.GCN_Branch_2(updated_features)

        # GCN result obtained by applying association matrix
        GCN_result = torch.matmul(self.Q, H2)  # self.norm_row_Q1 == self.Q
        GCN_result = GCN_result.permute([1, 0], 0)

        #Global Feature Prediction
        GCN_result = self.pooling(GCN_result)
        GCN_result = GCN_result.permute([1, 0], 0)
        GCN_result = self.linear_3(GCN_result)
        GCN_result = GCN_result.repeat(CNN_result_1.size(0), 1)
        GCN_result = F.softmax(GCN_result, dim=1)

        #Local-global feature collaborative classification
        out1 = torch.mul(CNN_result_1, GCN_result)
        out2 = torch.mul(CNN_result_2, GCN_result)
        out3 = torch.mul(CNN_result_3, GCN_result)
        return out1, out2, out3


