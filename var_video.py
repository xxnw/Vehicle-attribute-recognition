#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import pickle
import shutil
import random
import argparse

from darknet_util import *
from darknet import Darknet
from preprocess import prep_image, process_img, inp_to_image
from dataset import color_attrs, direction_attrs, type_attrs

import torch
import torchvision
import paramiko
import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.image import AxesImage
from scipy.spatial.distance import cityblock
from tqdm import tqdm

import keras.backend as K
import scipy.io

from devkit.car_label_cn import car_label
from utils_recog import load_model
# -------------------------------------
# for matplotlib to displacy chinese characters correctly
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

use_cuda = True  # True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(
    'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')

if use_cuda:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
print('=> device: ', device)

local_model_path = './checkpoints/epoch_39.pth'
local_car_cfg_path = './models/car.cfg'
local_car_det_weights_path = './models/car_540000.weights'

color_dict = {'Black': u'黑色', 'Blue': u'蓝色', 'Brown': u'棕色', 'Gray': u'灰色',
             'Green': u'绿色', 'Pink': u'粉色', 'Red': u'红色', 'White': u'白色', 'Yellow': u'黄色'}
direction_dict = {'Front': u'朝前', 'Rear': u'朝后'}
type_dict = {'passengerCar': u'客车', 'saloonCar': u'轿车', 'shopTruck': u'运货车', 'suv': u'SUV',
             'trailer': u'拖车', 'truck': u'卡车', 'van': u'商务车', 'waggon': u'旅行车'}


class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)  # 输出类别数
        # print('=> fc layer:\n', self.fc)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


# ------------------------------------- vehicle detection model
class Car_Classifier(object):
    """
    vehicle detection model mabager
    """

    def __init__(self,
                 num_cls,
                 model_path=local_model_path):
        """
        load model and initialize
        """

        # define model and load weights
        self.net = Cls_Net(num_cls=num_cls, input_size=224).to(device)
        # self.net = torch.nn.DataParallel(Net(num_cls=20, input_size=224),
        #                                  device_ids=[0]).to(device)
        self.net.load_state_dict(torch.load(model_path))
        print('=> vehicle classifier loaded from %s' % model_path)

        # set model to eval mode
        self.net.eval()

        # test data transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # split each label
        self.color_attrs = color_attrs
        print('=> color_attrs:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        print('=> direction attrs:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        print('=> type_attrs:\n', self.type_attrs)

    def get_predict(self, output):
        """
        get prediction from output
        """
        # get each label's prediction from output
        output = output.cpu()  # fetch data from gpu
        pred_color = output[:, :9]
        pred_direction = output[:, 9:11]
        pred_type = output[:, 11:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def pre_process(self, image):
        """
        image formatting
        :rtype: PIL.JpegImagePlugin.JpegImageFile
        """
        # image data formatting
        if type(image) == np.ndarray:
            if image.shape[2] == 3:  # turn all 3 channels to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 1:  # turn 1 channel to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # turn numpy.ndarray into PIL.Image
            image = Image.fromarray(image)
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            if image.mode == 'L' or image.mode == 'I':  # turn 8bits or 32bits into 3 channels RGB
                image = image.convert('RGB')

        return image

    def predict(self, img):
        """
        predict vehicle attributes by classifying
        :return: vehicle color, direction and type 
        """
        # image pre-processing
        img = self.transforms(img)
        img = img.view(1, 3, 224, 224)

        # put image data into device
        img = img.to(device)

        # calculating inference
        output = self.net.forward(img)

        # get result
        # self.get_predict_ce, return pred to host side(cpu)
        pred = self.get_predict(output)
        color_name = self.color_attrs[pred[0][0]]
        direction_name = self.direction_attrs[pred[0][1]]
        type_name = self.type_attrs[pred[0][2]]

        return color_name, direction_name, type_name


class Car_DC():
    def __init__(self,
                 src_path,
                 dst_path,
                 car_cfg_path=local_car_cfg_path,
                 car_det_weights_path=local_car_det_weights_path,
                 inp_dim=768,
                 prob_th=0.2,
                 nms_th=0.4,
                 num_classes=1):
        """
        model initialization
        """
        # super parameters
        self.inp_dim = inp_dim
        self.prob_th = prob_th
        self.nms_th = nms_th
        self.num_classes = num_classes
        self.dst_path = dst_path
        self.video_path = src_path

        # initialize vehicle detection model
        self.detector = Darknet(car_cfg_path)
        self.detector.load_weights(car_det_weights_path)
        # set input dimension of image
        self.detector.net_info['height'] = self.inp_dim
        self.detector.to(device)
        self.detector.eval()  # evaluation mode
        print('=> car detection model initiated.')

        # initiate multilabel classifier
        self.classifier = Car_Classifier(num_cls=19,
                                         model_path=local_model_path)
        
        #init car recognition
        self.img_width, self.img_height = 224, 224
        self.model = load_model()
        self.model.load_weights('models/model.96-0.89.hdf5')

        cars_meta = scipy.io.loadmat('devkit/cars_meta')
        class_names = cars_meta['class_names']  # shape=(1, 196)
        self.class_names = np.transpose(class_names)

    def cls_draw_bbox(self, output, orig_img):
        """
        1. predict vehicle's attributes based on bbox of vehicle
        2. draw bbox to orig_img
        """
        pt_1s = []
        pt_2s = []
        label_1 = []
        label_2 = []
        label_3 = []
        # 1
        for det in output:
            # rectangle points
            pt_1 = tuple(det[1:3].int())  # the left-up point
            pt_2 = tuple(det[3:5].int())  # the right down point
            pt_1s.append(pt_1)
            pt_2s.append(pt_2)

            # turn BGR back to RGB
            ROI = Image.fromarray(
                orig_img[pt_1[1]: pt_2[1],
                         pt_1[0]: pt_2[0]][:, :, ::-1])
            # ROI.show()

            # call classifier to predict
            car_color, car_direction, car_type = self.classifier.predict(ROI)
            label = str(car_color + ' ' + car_direction + ' ' + car_type)
            print('=> predicted label: ', label)
            label_1.append(str(car_color))
            label_2.append(str(car_direction))
            label_3.append(str(car_type))

        # 2
        color = (0, 255, 0)
        for i, det in enumerate(output):
            pt_1 = pt_1s[i]
            pt_2 = pt_2s[i]

            # draw bounding box
            cv2.rectangle(orig_img, pt_1, pt_2, color, thickness=2)

            # get str text size
            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            
            # Convert cv2 numpy array to PIL image
            cv2_im = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            # Draw a label with a chinese name in the filled box
            font = ImageFont.truetype('./font/simhei.ttf', int(txt_size[1] * 0.8), encoding="utf-8")#
            draw = ImageDraw.Draw(pil_im)

            # draw text background rect and text
            #car color
            pt_11 = pt_2[0], pt_1[1]
            pt_12 = pt_2[0] + txt_size[0] + 3, pt_1[1] + txt_size[1] + 3
            fname = color_dict[label_1[i]]
            draw.text((pt_11[0], pt_11[1]), ' 颜色: ' + fname, fill=(0, 255, 0), font=font)
    
            #car direction
            pt_21 = pt_2[0], pt_12[1]
            pt_22 = pt_2[0] + txt_size[0] + 3, pt_12[1] + txt_size[1] + 3
            fname = direction_dict[label_2[i]]
            draw.text((pt_21[0], pt_21[1]), ' 朝向: ' + fname, fill=(0, 255, 0), font=font)

            #car type
            pt_31 = pt_2[0], pt_22[1]
            fname = type_dict[label_3[i]]
            draw.text((pt_31[0], pt_31[1]), ' 车型: ' + fname, fill=(0, 255, 0), font=font)
                        
            # Convert PIL image to cv2 numpy array
            orig_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return orig_img

    def process_predict(self,
                        prediction,
                        prob_th,
                        num_cls,
                        nms_th,
                        inp_dim,
                        orig_img_size):
        """
        processing detections
        """
        scaling_factor = min([inp_dim / float(x)
                              for x in orig_img_size])  # W, H scaling factor
        output = post_process(prediction,
                              prob_th,
                              num_cls,
                              nms=True,
                              nms_conf=nms_th,
                              CUDA=True)  # post-process such as nms

        if type(output) != int:
            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  orig_img_size[0]) / 2.0  # x, w
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  orig_img_size[1]) / 2.0  # y, h
            output[:, 1:5] /= scaling_factor
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, orig_img_size[0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, orig_img_size[1])
        return output
    
    def car_recognition(self, output, orig_img):
        labels = []
        pt_1s = []
        pt_2s = []
        for det in output:
            # rectangle points
            pt_1 = tuple(det[1:3].int())  # the left-up point
            pt_2 = tuple(det[3:5].int())  # the right down point
            pt_1s.append(pt_1)
            pt_2s.append(pt_2)

            # turn BGR back to RGB
            ROI = Image.fromarray(
                orig_img[pt_1[1]: pt_2[1],
                         pt_1[0]: pt_2[0]][:, :, ::-1])
            img = cv2.cvtColor(np.asarray(ROI),cv2.COLOR_RGB2BGR)
            bgr_img = cv2.resize(img, (self.img_width, self.img_height), cv2.INTER_CUBIC)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = np.expand_dims(rgb_img, 0)
            preds = self.model.predict(rgb_img)
            prob = np.max(preds)
            class_id = np.argmax(preds)
            label = str(self.class_names[class_id][0][0])
            #print(label)
            labels.append(label)
            text = ('{}, {}'.format(self.class_names[class_id][0][0], prob))
            
        for i, det in enumerate(output):
            pt_1 = pt_1s[i]
            pt_2 = pt_2s[i]

            # get str text size
            txt_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            
            # Convert cv2 numpy array to PIL image
            cv2_im = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            # Draw a label with a chinese name in the filled box
            font = ImageFont.truetype('./font/simhei.ttf', int(txt_size[1] * 0.8), encoding="utf-8")#
            draw = ImageDraw.Draw(pil_im)
            # draw text background rect and text
            #car color
            pt_11 = pt_2[0], pt_1[1] + (txt_size[1] + 3) * 3
            pt_12 = pt_2[0] + txt_size[0] + 3, pt_11[1] + txt_size[1] + 3
            fname = car_label[labels[i]]
            draw.text((pt_11[0], pt_11[1]), ' 车类: ' + fname, fill=(0, 255, 0), font=font)
                       
            # Convert PIL image to cv2 numpy array
            orig_img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return orig_img            
            

    def detect_classify(self):
        """        
        detect and classify
        """
        #read and save video
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.dst_path, fourcc, 25.0, (960, 540))
        
        while(cap.isOpened()):
            # read image data
            ret, x = cap.read()
            re_img = cv2.resize(x, (960, 540), cv2.INTER_LINEAR)
            img = Image.fromarray(cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB))
            img2det = process_img(img, self.inp_dim)
            img2det = img2det.to(device)  # put image data to device

            # vehicle detection
            prediction = self.detector.forward(img2det, CUDA=True)

            # calculating scaling factor
            orig_img_size = list(img.size)
            output = self.process_predict(prediction,
                                          self.prob_th,
                                          self.num_classes,
                                          self.nms_th,
                                          self.inp_dim,
                                          orig_img_size)

            orig_img = cv2.cvtColor(np.asarray(
                img), cv2.COLOR_RGB2BGR)  # RGB => BGR
            if type(output) != int:
                orig_img = self.cls_draw_bbox(output, orig_img)
                orig_img = self.car_recognition(output, orig_img)
            out.write(orig_img)
        cap.release()
        out.release()
# -----------------------------------------------------------


if __name__ == '__main__':
    # ---------------------------- Car detect and classify
    DR_model = Car_DC(src_path='rtsp://admin:ai123456@192.168.160.101:554/h264/ch1/main/av_stream',
                       dst_path='./result/output_rtsp.avi')#./test/1.mp4'
    DR_model.detect_classify()