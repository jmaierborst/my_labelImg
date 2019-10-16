#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
import os
import csv
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from libs.constants import DEFAULT_ENCODING
#from data import predefined_classes

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class_names_coco = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class_names = ['person','mask','cap','glasses','glove','bloodyGlove','wrong']

class YOLOWriter:

    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def BndBox2YoloLine(self, box, classList=[]):
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

#        xcen = float((xmin + xmax)) / 2 / self.imgSize[1]
#        ycen = float((ymin + ymax)) / 2 / self.imgSize[0]
#
#        w = float((xmax - xmin)) / self.imgSize[1]
#        h = float((ymax - ymin)) / self.imgSize[0]

        # PR387
        boxName = box['name']
        if boxName not in classList:
            classList.append(boxName)

        classIndex = classList.index(boxName)

        return self.localImgPath, xmin, ymin, xmax, ymax, class_names[classIndex]

    def save(self, classList=[], targetFile=None):

        out_file = None #Update yolo .txt
        out_class_file = None   #Update class list .txt

        if targetFile is None:
            out_file = open(
            self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
            classesFile = os.path.join(os.path.dirname(os.path.abspath(self.filename)), "classes.txt")
            out_class_file = open(classesFile, 'w')

        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)
            classesFile = os.path.join(os.path.dirname(os.path.abspath(targetFile)), "classes.txt")
            out_class_file = open(classesFile, 'w')
            

        for box in self.boxlist:
            # BndBox2YoloLine returns image_path, xmin, ymin, xmax, ymax, className
            image_path, xmin, ymin, xmax, ymax, className = self.BndBox2YoloLine(box, classList)
            #csvRow = [image_path, xmin, ymin, xmax, ymax, className]
            
            image_path = image_path.replace('\\141.19.87.238','root')
            image_path = image_path.replace('\\','/')   #actually replaces \ with /
           
            out_file.write("{},{},{},{},{},{}\n" .format(image_path, xmin, ymin, xmax, ymax, className))
            
        for c in classList:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()


class YoloReader:

    def __init__(self, filepath, image, classListPath=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.filepath = filepath

        if classListPath is None:
            dir_path = os.path.dirname(os.path.realpath(self.filepath))
            self.classListPath = os.path.join(dir_path, "classes.txt")
        else:
            self.classListPath = classListPath

        # print (filepath, self.classListPath)

        classesFile = open(self.classListPath, 'r')
        self.classes = classesFile.read().strip('\n').split('\n')

        # print (self.classes)

        imgSize = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]

        self.imgSize = imgSize

        self.verified = False
        # try:
        self.parseYoloFormat()
        # except:
            # pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, xmin, ymin, xmax, ymax, difficult):

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))

    def yoloLine2Shape(self, classIndex, xcen, ycen, w, h):
        label = self.classes[int(classIndex)]

        xmin = max(float(xcen) - float(w) / 2, 0)
        xmax = min(float(xcen) + float(w) / 2, 1)
        ymin = max(float(ycen) - float(h) / 2, 0)
        ymax = min(float(ycen) + float(h) / 2, 1)

        xmin = int(self.imgSize[1] * xmin)
        xmax = int(self.imgSize[1] * xmax)
        ymin = int(self.imgSize[0] * ymin)
        ymax = int(self.imgSize[0] * ymax)

        return label, xmin, ymin, xmax, ymax

    def parseYoloFormat(self):
        bndBoxFile = open(self.filepath, 'r')
        for bndBox in bndBoxFile:
            filename, xmin, ymin, xmax, ymax, className = bndBox.split(',')
            className = className.replace('\n','')                              #
            classIndex = class_names.index(className)
            label = self.classes[int(classIndex)]
            # Caveat: difficult flag is discarded when saved as yolo format.
            self.addShape(label, xmin, ymin, xmax, ymax, False)
