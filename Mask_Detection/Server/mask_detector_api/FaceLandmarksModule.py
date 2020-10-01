#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import cv2
import time
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
from matplotlib import pyplot as plt
import imutils

# In[29]:


class Landmark_Extractor:

    @staticmethod
    def init_plugin(device,cpu_extension_path = ""):
        plugin = IEPlugin(device = device)
        if device == "CPU" and cpu_extension_path!="":
            plugin.add_cpu_extension(cpu_extension_path)
        return plugin
    
    def load_net(self,model_xml,model_bin,plugin,num_requests = 2):
        net = IENetwork(model = model_xml, weights= model_bin)
        not_supported_layers = []

        supported_layers = plugin.get_supported_layers(net)
        net_layers = net.layers

        for layer in supported_layers:
            if not layer in supported_layers:
                not_supported_layers.append(layer)
        
        if len(not_supported_layers)>0:
            print("WARNING: None supported layers detected, please review network artchtecture before continuing...")
            print(not_supported_layers)
        else:
            print("INFO: All network layers are supported.")
        
        self.exec_net = plugin.load(network=net,num_requests=num_requests)
    
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))

        self.input_shape = net.inputs[self.input_blob].shape
        self.output_shape = net.outputs[self.output_blob].shape

        print("Input Shape: {}".format(self.input_shape))
        print("Output Shape: {}".format(self.output_shape))
    
    
    def extractLandmarks(self,img):
        img = cv2.resize(img,(self.input_shape[2],self.input_shape[3]))
        img = img.transpose((2, 0, 1))  
        img = np.expand_dims(img,0)
    
        pred = self.exec_net.infer(inputs = {self.input_blob:img})
        print(pred)
        landmarks = pred[self.output_blob][0]
        landmarks_pairs = []
        for i in range(0,len(landmarks),2):
            landmarks_pairs.append((landmarks[i],landmarks[i+1]))
        return landmarks_pairs
    
    def calculate_reference_landmarks(self,img):
        reference_landmarks = [(0.31556875000000000*img.shape[1], 0.4615741071428571*img.shape[0]),
 (0.68262291666666670*img.shape[1], 0.4615741071428571*img.shape[0]),
 (0.50026249999999990*img.shape[1], 0.6405053571428571*img.shape[0]),
 (0.34947187500000004*img.shape[1], 0.8246919642857142*img.shape[0]),
 (0.65343645833333330*img.shape[1], 0.8246919642857142*img.shape[0])]
        return reference_landmarks
    
    def process_landmarks(self,landmarks,img):
        left_eye_points = tuple(np.mean((landmarks[0],landmarks[1]),axis=0))
        right_eye_points =tuple(np.mean((landmarks[2],landmarks[3]),axis=0))
        nose_tip_points =  landmarks[4]
        left_lip_corner_points = landmarks[8]
        right_lip_corner_points = landmarks[9]

        extracted_landmarks = [left_eye_points,right_eye_points,nose_tip_points,left_lip_corner_points
                               ,right_lip_corner_points]
        extracted_landmarks = list(map(lambda x:(x[0]*img.shape[1],x[1]*img.shape[0]),extracted_landmarks))
        return extracted_landmarks
    
    
    def procrustes2(self,X, Y, scaling=True, reflection='best'):
        """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

        """

        n,m = X.shape
        ny,my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()

    # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

    # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection is not 'best':

        # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

        # optimum scaling of Y
            b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2

        # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
        if my < m:
            T = T[:my,:]
        c = muX - b*np.dot(muY, T)

    #transformation values 
        tform = {'rotation':T, 'scale':b, 'translation':c}

        return d, Z, tform
    
    
    def transform_face(self,img,landmarks,reference_landmarks,visualize = False):

        extracted_landmarks = self.process_landmarks(landmarks,img)
        d,Z_pts,Tform = self.procrustes2(np.asarray(reference_landmarks),np.asarray(extracted_landmarks))

        R = np.eye(3)
        R[0:2,0:2] = Tform['rotation']
        S = np.eye(3) * Tform['scale'] 
        S[2,2] = 1
        t = np.eye(3)
        t[0:2,2] = Tform['translation']
        transform_mat = np.dot(np.dot(R,S),t.T).T

        img = cv2.warpAffine(img,transform_mat[0:2,:],(img.shape[1],img.shape[0]))
    
        if visualize:
            img_vis = img.copy()
            for (x,y) in reference_landmarks:
                x = int(x)
                y = int(y)
                cv2.circle(img_vis,(x,y),int(img_vis.shape[0]/30),(0,255,0),-1)

            plt.imshow(cv2.cvtColor(img_vis,cv2.COLOR_BGR2RGB))
            plt.show()
        return img
    
    
    def prepare_face(self,img,visualize = False,verbose=False):
        if verbose:
            start = time.time()
            landmarks = self.extractLandmarks(img)
            end = time.time()
            print("Landmarks extracted in {} milliesconds.".format(int((end-start)*1000)))
            start = time.time()
            T = self.transform_face(img,landmarks,self.calculate_reference_landmarks(img),visualize)
            end = time.time()
            print("Face re-alligned in {} milliesconds.".format(int((end-start)*1000)))
        else:
            landmarks = self.extractLandmarks(img)
            T = self.transform_face(img,landmarks,self.calculate_reference_landmarks(img),visualize)
        return T
