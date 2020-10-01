# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os

from collections import deque
from collections import Counter
import imutils
import _thread as thread
import uuid
from FaceDetectionModule import Face_Detector
from FaceGazeModule import GazeEstimator
import glob

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    
    
maxQueueLen = 20
developer_mode = True
device = "CPU"
cpu_extension_path = "cpu_extension_avx2.dll"
detect_thresh = 0.7
min_rect_area = 4000
turn_angle_thresh = 15


model_xml = r"Models/openvino/face-detection/FP32/face-detection-retail-0004.xml"
model_bin = r"Models/openvino/face-detection/FP32/face-detection-retail-0004.bin"

plugin = Face_Detector.init_plugin(device,cpu_extension_path)
faceDetector = Face_Detector()
faceDetector.load_net(model_xml,model_bin,plugin)


model_xml = r"Models/openvino/gaze-estimation/FP32/head-pose-estimation-adas-0001.xml"
model_bin =r"Models/openvino/gaze-estimation/FP32/head-pose-estimation-adas-0001.bin"
gazeEstimator = GazeEstimator()
gazeEstimator.load_net(model_xml,model_bin,plugin)


prototxtPath = "Models/deploy.prototxt"
weightsPath = "Models/fd_ov_m.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = load_model("models/mask_detector.model")
skip_frames = 0


@csrf_exempt
def detect(request):
    data = {"success": False}
    if request.method == "POST":
        if request.FILES.get("image", None) is not None:
            image = _grab_image(stream=request.FILES["image"])
        else:
            url = request.POST.get("url", None)
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            image = _grab_image(url=url)
            rects,_ = faceDetector.detectFaces(image,detect_thresh)
            faces = {}
            faces_imgs = []
            mask_status = []
            for i,(startX, startY, endX, endY) in enumerate(rects):
                face = image[startY:endY, startX:endX]
                if face is None or len(face) == 0:
                    continue
                rect_width = endX - startX
                rect_height = endY - startY
                faces[i] = rect_width * rect_height
                faces_imgs.append(face)
            faces = {k: v for k, v in sorted(faces.items(), key=lambda item: item[1],reverse = True)}
            if len(faces) > 0 and faces[list(faces.keys())[0]] > min_rect_area:
                ix = list(faces.keys())[0]
                face = faces_imgs[ix]
                if gazeEstimator.detectFaces(face)[1] >turn_angle_thresh:
                    return -1,None,None
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                (mask, withoutMask) = model.predict(face)[0]
                label = "Mask" if mask > withoutMask else "No Mask"
                data.update({"face_index": ix, "face": rects[ix],"label" : label ,"success": True})
            else:
                data.update({"face_index": -1, "face": None,"label" : None ,"success": False})
    return JsonResponse(data)


@csrf_exempt
def test(request):
    return JsonResponse({"name":"Ahmed Omar","Age":33,"path":model_xml})
    
def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:    
        # if the URL is not None, then download the image
        if url is not None:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # return the image
    return image