from django.shortcuts import render

# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import base64

from collections import deque
from collections import Counter
import imutils
import uuid
from FaceDetectionModule import Face_Detector
from FaceGazeModule import GazeEstimator
import glob
import threading

from django.middleware import csrf
from django.http import HttpResponse

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from django import forms
import time

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
    
    
maxQueueLen = 20
developer_mode = True
device = "CPU"
cpu_extension_path = "cpu_extension_avx2.dll"
detect_thresh = 0.7
min_rect_area = 15000
turn_angle_thresh = 100


model_xml = r"Models/openvino/face-detection/FP32/face-detection-retail-0004.xml"
model_bin = r"Models/openvino/face-detection/FP32/face-detection-retail-0004.bin"

plugin = Face_Detector.init_plugin(device,cpu_extension_path)
faceDetector = Face_Detector()
faceDetector.load_net(model_xml,model_bin,plugin)


model_xml = r"Models/openvino/gaze-estimation/FP32/head-pose-estimation-adas-0001.xml"
model_bin =r"Models/openvino/gaze-estimation/FP32/head-pose-estimation-adas-0001.bin"
gazeEstimator = GazeEstimator()
gazeEstimator.load_net(model_xml,model_bin,plugin)

model = load_model("models/mask_detector_v3.model")
skip_frames = 0



############################
camera_url = "rtsp://sda:Connectiv123!@10.0.1.41/Streaming/channels/101"
lock = threading.Lock()
time.sleep(2)
outputFrame = None
outputFace = None

c_left=0.172
c_right= 0.23
c_top=0.152
c_bottom = 0

############################


def start_capture():
    global camera_url
    global outputFrame,lock
    video = cv2.VideoCapture(camera_url)
    time.sleep(2)
    print("Starting Camera...")
    while True:
        ret, frame = video.read()
        if ret == False or len(frame) == 0:
            print("Failed to capture frame, skipping...")
            continue
     
               
                
#         w = frame.shape[1]
#         h = frame.shape[0]
        
#         print((c_top * h , h - (c_bottom * h)))
#         print((c_left * w , w-(c_right * w)))
        
#         frame = frame[int(c_top * h) : int(h - (c_bottom * h)),int( c_left * w) :int( w-(c_right * w))]
        
        cv2.imshow("Camera Stream",frame)
        
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        with lock:
            if frame is not None:
                outputFrame = frame.copy()
            
t = threading.Thread(target=start_capture)
t.daemon = True
t.start()
            
def get_or_create_csrf_token(request):
    token = request.META.get('CSRF_COOKIE', None)
    if token is None:
        token = csrf._get_new_csrf_key()
        request.META['CSRF_COOKIE'] = token
    request.META['CSRF_COOKIE_USED'] = True
    return token


def generate():
    global outputFrame, lock
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')


@csrf_exempt
def stream_frame():
    return HttpResponse(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")
    
@csrf_exempt
def detect_from_stream(request):
    global min_rect_area
    global turn_angle_thresh,lock
    global outputFrame
    #global outputFace
    image = None

    data = {"success": False}
    print("starting")
    if request.method == "GET":        
        with lock:
            if outputFrame is not None:
                image = outputFrame.copy()
            else:
                print("Output frame is none.")
                image = None
            
        if image is None or len(image) == 0:
            return JsonResponse(data)

        print("Image Grabbed!")
        print(image.shape)
        #cv2.imwrite("image.png",image);
        
        w = image.shape[1]
        h = image.shape[0]
        
     #   image = image[c_top * h : h - (c_bottom * h), c_left * w : w-(c_right * w)]
                
        image = image[int(c_top * h) : int(h - (c_bottom * h)),int( c_left * w) :int( w-(c_right * w))]
        
        rects,_ = faceDetector.detectFaces(image,detect_thresh)
        faces = {}
        faces_imgs = []
        mask_status = []
        for i,(startX, startY, endX, endY) in enumerate(rects):
            print("Faces detected")
            face = image[startY:endY, startX:endX]
            if face is None or len(face) == 0:
                continue
            print("Face extracted")
            rect_width = endX - startX
            rect_height = endY - startY
            #print("Face Area = " + str(rect_width * rect_height))
            faces[i] = rect_width * rect_height
            faces_imgs.append(face)
        faces = {k: v for k, v in sorted(faces.items(), key=lambda item: item[1],reverse = True)}
        print("faces arranged")
        
        if len(faces) > 0 and faces[list(faces.keys())[0]] > min_rect_area:
            print(faces[list(faces.keys())[0]])
            print("Processing face area...")
            ix = list(faces.keys())[0]
            face = faces_imgs[ix]
         #   outputFace = face
          #  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
           # MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
           # cv2.imwrite(os.path.join(MEDIA_ROOT,"1.png"),face)
            print("Head Angle = " + str(gazeEstimator.detectFaces(face)[1]))
            print(turn_angle_thresh)
            if abs(gazeEstimator.detectFaces(face)[1]) >turn_angle_thresh:
                return JsonResponse(data)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            print("Predicting...")
            (mask, withoutMask) = model.predict(face)[0]
            print((mask, withoutMask))
            print("Predicted")
            label = "Mask" if mask > withoutMask else "No Mask"
            data.update({"face_index": ix, "face": rects[ix],"label" : label ,"success": True})
        else:
            data.update({"face_index": -1, "face": None,"label" : None ,"success": False})
            
    print(data)
    return JsonResponse(data)

# @csrf_exempt
# def get_latest_face(request):
#     global outputFace
#     if outputFace is  None:
#         return JsonResponse({"success" : False, "result" : None})
#     else:
#         _, im_arr = cv2.imencode('.jpg', outputFace)
#         im_bytes = im_arr.tobytes()
#         im_b64 = base64.b64encode(im_bytes)
#         return JsonResponse({"success" : True, "result" : im_b64.decode()})

@csrf_exempt
def detect(request):
    global min_rect_area
    global turn_angle_thresh
    
    try:
        data = {"success": False}
        print("starting")
        if request.method == "POST":
            if request.FILES.get("image", None) is not None:
                image = _grab_image(stream=request.FILES["image"])
                print("Image Grabbed!")
                print(image.shape)
               # cv2.imwrite("image.png",image);
               # min_rect_area = int(request.POST.get("area", None))
                min_rect_area = 30000
                turn_angle_thresh = 60
               # turn_angle_thresh = int(request.POST.get("turn_angle_thresh", None))
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
                print("Faces detected")
                face = image[startY:endY, startX:endX]
                if face is None or len(face) == 0:
                    continue
                print("Face extracted")
                rect_width = endX - startX
                rect_height = endY - startY
                #print("Face Area = " + str(rect_width * rect_height))
                faces[i] = rect_width * rect_height
                faces_imgs.append(face)
            faces = {k: v for k, v in sorted(faces.items(), key=lambda item: item[1],reverse = True)}
            print("faces arranged")
            if len(faces) > 0 and faces[list(faces.keys())[0]] > min_rect_area:
                print(faces[list(faces.keys())[0]])
                print("Processing face area...")
                ix = list(faces.keys())[0]
                face = faces_imgs[ix]
                print("Head Angle = " + str(gazeEstimator.detectFaces(face)[1]))
                print(turn_angle_thresh)
                if abs(gazeEstimator.detectFaces(face)[1]) >turn_angle_thresh:
                    return JsonResponse(data)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                print("Predicting...")
                (mask, withoutMask) = model.predict(face)[0]
                print((mask, withoutMask))
                print("Predicted")
                label = "Mask" if mask > withoutMask else "No Mask"
                data.update({"face_index": ix, "face": rects[ix],"label" : label ,"success": True})
            else:
                data.update({"face_index": -1, "face": None,"label" : None ,"success": False})
    except:
        return JsonResponse({"success": False})
            
    print(data)
    return JsonResponse(data)


@csrf_exempt
def test(request):
    return JsonResponse({"name":"Ahmed Omar","Age":33,"path":model_xml})


@csrf_exempt
def validate_cert(request):
    return JsonResponse({"cert_status":"Validated"})
    
def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:    
        # if the URL is not None, then download the image
        if url is not None:
            print("downloading " + url)
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
