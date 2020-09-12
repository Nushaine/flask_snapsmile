import subprocess
import torch
import glob
import os
import shutil


def getImages(path):
    imagesList = []
    for image in os.listdir(path):
        if os.path.isfile(os.path.join(path, image)):
            imagesList.append(image)
    return imagesList


def copyOutputs(source, destination):
    for file in os.listdir(source):
        try:
            shutil.copy(os.path.join(source, file), destination)
        except:
            continue


def testModel():
    print("HELO")
    print(torch.__version__)
    # changes directories to /yolov5
    first = subprocess.call('dir', shell=True, cwd='yolov5')
    # test model
    second = subprocess.call(
        'python detect.py --weights weights/weigths_v2.pt --conf 0.80 --source ../inputs', shell=True, cwd='yolov5')
    # move_outputs_to_static = copyOutputs('yolov5/inference/output', 'static/outputs')
    imageList = getImages('yolov5/inference/output')
    return imageList
