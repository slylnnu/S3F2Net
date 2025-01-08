import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import cv2
import math


def LSC_superpixel(I, nseg):
    '''
    Superpixel Graph Construction Using LSC (Linear Spectral Clustering)
    '''
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SegmentsLabelProcess(labels):
    '''
    Perform post-processing on the labels to prevent the occurrence of discontinuous labels.
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SEEDS(object):
    def __init__(self, LiDAR, n_segments, num_levels=2, prior=1, histogram_bins=5, num_iterations=4):
        """
               Perform SEEDS superpixel segmentation on the image.

               Parameters:
                   I (numpy.ndarray): Input image (height x width x channels).
                   n_segments (int): Number of desired superpixels.
                   num_levels (int, optional): Number of pyramid levels for segmentation. Default is 2.
                   prior (int, optional): Prior for segmentation. Default is 1.
                   histogram_bins (int, optional): Number of histogram bins for color quantization. Default is 5.
                   num_iterations (int, optional): Number of iterations for SEEDS algorithm. Default is 4.

               Returns:
                   numpy.ndarray: Segmented image with superpixel labels.
               """
        self.n_segments = n_segments
        self.num_levels = num_levels
        self.prior = prior
        self.histogram_bins = histogram_bins
        self.num_iterations = num_iterations
        self.data=LiDAR

    def SEEDS_superpixel(self,I):
        I_new = np.array(I[:, :, 0:3], np.float32).copy()  # Convert to float32
        height, width, channels = I_new.shape

        # Create SEEDS superpixel object
        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(self.n_segments),
                                                   num_levels=self.num_levels, prior=self.prior,
                                                   histogram_bins=self.histogram_bins)

        # Iterate for the given number of iterations
        seeds.iterate(I_new, self.num_iterations)

        # Get the segmented labels
        segments = seeds.getLabels()

        return segments

    def get_Q_and_S_and_Segments(self,img):
        '''
        Compute the superpixel 'segments' and the associated matrix 'Q'
        '''
        (h, w, d) = img.shape
        segments = self.SEEDS_superpixel(img)

        #SLIC（Simple Linear Iterative Clustering）
        # segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_iter=self.max_iter,
        #                 convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
        #                 min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        # Check if the superpixel labels are consecutive; otherwise, correct them.
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        # Display superpixel image
        out = mark_boundaries(img[:, :, 0], segments)
        plt.figure()
        plt.imshow(out)
        plt.show()

        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            S[i] = superpixel
            Q[idx, i] = 1

        self.S = S
        self.Q = Q

        return Q, S, self.segments

    def get_A(self, sigma: float):
        '''
        Determine the adjacency matrix based on segments
        '''
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        return A


class Superpixel_Seg(object):
    def __init__(self, scale):
        self.scale=scale
    def SGB(self,data):
        # Superpixel Graph Builder (SGB)
        height, width = data.shape[:2]
        n_segments_init = height * width / self.scale
        print(f"Initial number of segments: {n_segments_init}")
        myseeds = SEEDS(data, n_segments_init, num_levels=2, prior=1, histogram_bins=5, num_iterations=4)
        Q, S, Segments = myseeds.get_Q_and_S_and_Segments(data)
        A = myseeds.get_A(sigma=10)
        return Q, S, A, Segments

