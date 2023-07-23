#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import numpy as np
import subprocess

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import cv2
import time
import argparse
from os import path as osp
import glob
from shutil import rmtree
from parse import parse_tfrecord
import threading

import psutil

if __name__=='__main__':
    # TODO argparse 적용
    download_dir = './tmp'
    version = '1_4_2'
    split   = 'validation'

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    dir_url = 'gs://waymo_open_dataset_v_{version}/individual_files/{split}'.format(version=version, split=split)
    command = 'gsutil list {url}'.format(url=dir_url)
    files = subprocess.check_output(command, shell=True, encoding='utf-8').split('\n')
    files = [fn for fn in files if fn != '']

    fn_whitelist = osp.join(download_dir,'whitelist.txt')
    fn_blacklist = osp.join(download_dir,'blacklist.txt')
    whitelist = []
    try:
        with open(fn_whitelist,"r") as f:
            for line in f.readlines():
                whitelist.append(line.strip())
    except FileNotFoundError:
        pass
    whitelist = [url for url in whitelist if url != '']

    blacklist = []
    try:
        with open(fn_blacklist,"r") as f:
            for line in f.readlines():
                blacklist.append(line.strip())
    except FileNotFoundError:
        pass
    blacklist = [url for url in blacklist if url != '']

    for i, url in enumerate(files):
        if len(url) == 0:
            continue
        if url in whitelist:
            print("Pass file in whitelist %s" % url)
            continue
        if url in blacklist:
            print("Pass file in blacklist %s" % url)
            continue
        print("%d/%d" %(i,len(files)) )
        base = osp.basename(url)
        fn   = osp.join(download_dir, base)
        flag = os.system('gsutil cp {url} {download_dir}'.format(url=url, download_dir=download_dir) )
        assert flag == 0, 'Failed to download segment %s'%url
        if parse_tfrecord(fn, use_inpaint=True, verbose=True):
            with open(fn_whitelist,"a") as f:
                f.write(url+"\n")
        else:
            with open(fn_blacklist,"a") as f:
                f.write(url+"\n")
        os.remove(fn) # Remove after converting 

        disk_usage = psutil.disk_usage('/')
        remaining_space = disk_usage.free / (1024**3)  # Convert to gigabytes
        if remaining_space < 20. : #[GB]
          print("Stop download&parsing for space shortage")
          break
    print("done")
