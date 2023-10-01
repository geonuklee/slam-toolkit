#!/usr/bin/python2
#-*- coding:utf-8 -*-
import cv2
import numpy as np
import struct

colors = (
  (0,180,0),
  (0,100,0),
  (255,0,255),
  (100,0,255),
  (100,0,100),
  (0,0,180),
  (0,0,100),
  (255,255,0),
  (100,255,0),
  (100,100,0),
  (100,0,0),
  (0,255,255),
  (0,100,255),
  (0,255,100),
  (0,100,100)
)

def GetColoredLabel(marker, text=False):
    dst = np.zeros((marker.shape[0],marker.shape[1],3), dtype=np.uint8)
    uniq = np.unique(marker)
    n= len(colors)
    for u in uniq:
        if u <= 0:
            continue
        part = marker==u
        color = colors[u%n]
        dst[part] = color
        if not text:
            continue
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats( (part).astype(np.uint8) )
        color = (255-color[0], 255-color[1], 255-color[2])
        for i, (x0,y0,w,h,s) in enumerate(stats):
            if w == marker.shape[1] and h == marker.shape[0]:
                continue
            cp = centroids[i].astype(np.int)
            msg = '%d'%u
            w,h = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN,1.5,2)[0]
            cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(255,255,255),-1)
            cv2.rectangle(dst,(cp[0]-2,cp[1]+5),(cp[0]+w+2,cp[1]-h-5),(100,100,100),1)
            cv2.putText(dst, msg, (cp[0],cp[1]), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    return dst

def ReadRawImage(file_path, dtype=np.int32):
    # Read the raw image file
    with open(file_path, 'rb') as file:
        # Read the header information
        rows = struct.unpack("<i", file.read(4))[0]
        cols = struct.unpack("<i", file.read(4))[0]
        channels = struct.unpack("<i", file.read(4))[0]
        data_size = struct.unpack("<i", file.read(4))[0]
        # Read the binary data
        binary_data = file.read(data_size)
    return np.frombuffer(binary_data, dtype=dtype).reshape(rows, cols)

def GetMarkerCenters(given_marker):
    marker = given_marker.copy()
    marker[:,0] = 0
    marker[:,-1] = 0
    marker[0,:] = 0
    marker[-1,:] = 0
    centers = {}
    distances = {}
    for marker_id in np.unique(marker):
        if marker_id == 0:
            continue
        part = marker==marker_id
        dist_part = cv2.distanceTransform( part.astype(np.uint8),
                distanceType=cv2.DIST_L2, maskSize=5)
        loc = np.unravel_index( np.argmax(dist_part,axis=None), marker.shape)
        centers[marker_id] = (loc[1],loc[0])
        distances[marker_id] = dist_part[loc]
    #dst = GetColoredLabel(marker)
    #for marker_id, cp in centers.items():
    #    cv2.putText(dst, "%d" % marker_id, (cp[0],cp[1]), cv2.FONT_HERSHEY_SIMPLEX, 1., (0), 2)
    return centers, distances

