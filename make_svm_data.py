# -*- coding: utf-8 -*-
import cv2
import sys
import os
import commands
import numpy as np

data_dir = '/tmp/svm'

def cmd(cmd):
    return commands.getoutput(cmd)

def extract_color(src, h_th_low, h_th_up, s_th, v_th):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if h_th_low > h_th_up:
        ret, h_dst_1 = cv2.threshold(h, h_th_low, 255, cv2.THRESH_BINARY) 
        ret, h_dst_2 = cv2.threshold(h, h_th_up,  255, cv2.THRESH_BINARY_INV)

        dst = cv2.bitwise_or(h_dst_1, h_dst_2)
    else:
        ret, dst = cv2.threshold(h,   h_th_low, 255, cv2.THRESH_TOZERO) 
        ret, dst = cv2.threshold(dst, h_th_up,  255, cv2.THRESH_TOZERO_INV)
        ret, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)

    ret, s_dst = cv2.threshold(s, s_th, 255, cv2.THRESH_BINARY)
    ret, v_dst = cv2.threshold(v, v_th, 255, cv2.THRESH_BINARY)
    dst = cv2.bitwise_and(dst, s_dst)
    dst = cv2.bitwise_and(dst, v_dst)
    return dst

def average_color(src):
    bgr_img = src
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    average_bgr = [0,0,0]
    average_hsv = [0,0,0]

    # measure average of RGB color
    for i in range(3):
        extract_img = bgr_img[:,:,i]
        extract_img = extract_img[extract_img>0]
        average_bgr[i] = np.average(extract_img)

    # measure average of HSV color
    for i in range(3):
        extract_img = hsv_img[:,:,i]
        extract_img = extract_img[extract_img>0]
        average_hsv[i] = np.average(extract_img)

    rgb_value = [average_bgr[2], average_bgr[1], average_bgr[0]]
    hsv_value = [average_hsv[0], average_hsv[1], average_hsv[2]]

    return rgb_value, hsv_value

dirs = cmd("ls "+sys.argv[1])
labels = dirs.splitlines()

if os.path.exists(data_dir):
    cmd("rm  -rf "+data_dir)

# make directories
os.makedirs(data_dir)

pwd = cmd('pwd')

svm_data = open(data_dir + '/svm.tsv','w')

for label in labels:
    workdir = pwd+"/"+sys.argv[1]+"/"+label
    imageFiles = cmd("ls "+workdir+"/*.jpg")
    images = imageFiles.splitlines()
    # print(label)
    length = len(images)
    for image in images:
        print(image)
        # image processing
        input_img = cv2.imread(image)
        msk_img = extract_color(input_img, 100, 280, 20, 20)
        filter_img = cv2.bitwise_and(input_img, input_img, mask = msk_img)
        rgb_value, hsv_value = average_color(filter_img)
        # write tsv data
        svm_data.write(label+"\t"+str(rgb_value[0])+"\t"+str(rgb_value[1])+"\t"+str(rgb_value[2])
                            +"\t"+str(hsv_value[0])+"\t"+str(hsv_value[1])+"\t"+str(hsv_value[2])+"\n")

svm_data.close()
