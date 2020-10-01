#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import cv2
import time
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
from matplotlib import pyplot as plt
import imutils
print("Imported Python modules.")


# In[54]:


class GazeEstimator:
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
        self.output_blob_y = "angle_y_fc"
        self.output_blob_p = "angle_p_fc"
        self.output_blob_r = "angle_r_fc"

        self.input_shape = net.inputs[self.input_blob].shape
        self.output_shape = net.outputs[self.output_blob_y].shape

        print("Input Shape: {}".format(self.input_shape))
        print("Output Shape: {}".format(self.output_shape) + "*3 for (Y,P,R)")
        
    
    def detectFaces(self,img):
        img = cv2.resize(img,(self.input_shape[2],self.input_shape[3]))
        img = img.transpose((2, 0, 1))  
        img = np.expand_dims(img,0)

        pred = self.exec_net.infer(inputs = {self.input_blob:img})
        
        return pred[self.output_blob_y][0][0],pred[self.output_blob_p][0][0],pred[self.output_blob_r][0][0]

