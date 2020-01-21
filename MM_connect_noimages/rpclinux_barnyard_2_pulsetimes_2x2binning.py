import rpyc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1,0'

import threading as t
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':1}))
    
import keras
import keras.backend as K
K.set_session(sess)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import skimage.io
from glob import glob
import random
import time 
import pandas as pd
import scipy.ndimage
import umap
import warnings
import io
import base64

sys.path.append('../DSB_2018-master/')
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model_inference as modellib
from mrcnn.model import log
from mrcnn import my_inference
#import multiprocessing as mp

class LinuxService(rpyc.Service):
    class exposed_ImageAnalysis(object):
        def __init__(self, callback):

            self.callback = rpyc.async_(callback)   # make the callback async
            return        

        def exposed_finalize(self):
            
            #Save data as dataframes
            image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated', 'SegmentationTime', 'TotalTime'])
            cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Cell_Size', 'Boundary_Cell', 'Activated'])
            image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
            cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
            return

        def exposed_run_pipeline_on_image(self, image_name, img):
            
            image_name = os.path.splitext(image_name)[0]
            global image_number
            image_dir = os.path.join(output_dir, str(image_number))
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
                os.mkdir(os.path.join(image_dir, 'cell_images'))
                os.mkdir(os.path.join(image_dir, 'masks'))
            #import image
            img = np.frombuffer(base64.decodebytes(img), np.uint8).reshape((image_height,image_width)) #expect only the dendra channel!
            skimage.io.imsave(os.path.join(image_dir,'image.png'), img)

            #normalize and save
            img_norm = self.normalize_image(img) #only dendra channel

            #analyze
            output_mask = self.analyze_image(image_name, img_norm, image_dir)

            #save output mask
            skimage.io.imsave(os.path.join(image_dir, 'mask.png'), output_mask)

            #encode base64 and output
            if not output_mask.flags.c_contiguous:
                output_mask = output_mask.copy(order='C')
            return base64.b64encode(output_mask)

            #t.Thread(target=self.save_files, args=(image_name, img, output_mask)).start()
            #mp.Process(target=self.save_files, args=(image_name, img, output_mask)).start()

        def normalize_image(self, img):
            
            #Normalize images
            percentile = 99.9;
            high = np.percentile(img, percentile);
            low = np.percentile(img, 100-percentile);

            img = np.minimum(high, img);
            img = np.maximum(low, img);

            img = (img - low) / (high - low); # gives float64, thus cast to 8 bit later
            img = skimage.img_as_ubyte(img);

            # Make a RGBA-channel color image
            img_norm = np.stack([img, img, img], axis=-1);
            return img_norm;

        def analyze_image(self, image_name, norm_dendra_image, image_dir):

            # Catch warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Mold image
                start_time = time.time()
                molded_image, window, scale, padding, crop = utils.resize_image(norm_dendra_image, min_dim=inference_config.IMAGE_MIN_DIM, min_scale=inference_config.IMAGE_MIN_SCALE, max_dim=inference_config.IMAGE_MAX_DIM, mode=inference_config.IMAGE_RESIZE_MODE)
                active_class_ids = [1,1]
                image_metas = modellib.compose_image_meta(0, (image_height,image_width,3), molded_image.shape, window, scale, active_class_ids)

                # Run object detection
                results = model.detect_molded(np.expand_dims(molded_image, 0), np.expand_dims(image_metas, 0), verbose=1)
                end_time1 = time.time()
                segmentation_time = end_time1-start_time
                n_cells = len(results[0]['class_ids'])
                print('Number of cells: ' + str(n_cells))
                print('Segmentation time in ms: ' + str(segmentation_time*1000))

                # Compute crop dimensions, cell dimensions, centroids
                list_of_crops = []
                cell_y_min = results[0]['rois'][:,0]
                cell_y_max = results[0]['rois'][:,2]
                crop_y_min = np.maximum(0,cell_y_min-expand_pixels)
                crop_y_max = np.minimum(image_metas[4],cell_y_max+expand_pixels)

                cell_x_min = results[0]['rois'][:,1]
                cell_x_max = results[0]['rois'][:,3]
                crop_x_min = np.maximum(0,cell_x_min-expand_pixels)
                crop_x_max = np.minimum(image_metas[5],cell_x_max+expand_pixels)

                cell_y_dim = cell_y_max - cell_y_min 
                cell_x_dim = cell_x_max - cell_x_min
                crop_y_dim = crop_y_max - crop_y_min
                crop_x_dim = crop_x_max - crop_x_min

                cell_y_centroid = np.int_((cell_y_min + cell_y_max)/2-padding[0][1])
                cell_x_centroid = np.int_((cell_x_min + cell_x_max)/2-padding[1][1])

                cell_size = np.zeros(n_cells, dtype='uint32')
                boundary_cell = (((cell_x_min - padding[1][1]) <= boundary_size) | ((cell_x_max - padding[1][1]) >= image_metas[2] - boundary_size)) | (((cell_y_min - padding[0][1]) <= boundary_size) | ((cell_y_max - padding[0][1]) >= image_metas[1] - boundary_size))

                # Perform crops of images to put into classifier
                for i in range(n_cells):

                    # Crop cell in form of mask into crops of crop_size x crop_size
                    cell_size[i] = sum(sum(results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i]))
                    crop = molded_image[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],0]
                    mask = results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i]
                    crop = crop * mask
                    y_pad = (crop_size - crop_y_dim[i])//2
                    y_odd = crop_y_dim[i] % 2
                    x_pad = (crop_size - crop_x_dim[i])//2
                    x_odd = crop_x_dim[i] % 2
                    crop = np.pad(crop, ((max(0,y_pad),max(0,y_pad + y_odd)), (max(0,x_pad),max(0,x_pad + x_odd))), mode='constant')[max(0,-y_pad):max(0,-y_pad)+crop_size, max(0,-x_pad):max(0,-x_pad)+crop_size]
                    
                    # Normalize cell, then convert cell to tensor and move to gpu
                    if normalize_crops is True:
                        list_of_crops.append(torch.from_numpy(np.moveaxis(self.normalize_image(crop), -1, 0)))
                    else:
                        list_of_crops.append(torch.from_numpy(np.stack([crop, crop, crop], axis=0)))

                    # Save crops
                    skimage.io.imsave(os.path.join(image_dir,'cell_images',str(i)+'.png'), crop)
            
                # Perform classifier predictions on gpu
                if n_cells > 0:
                    predictions = np.concatenate([np.argmax(learn.pred_batch(ds_type=DatasetType.Test, batch=(normalize(torch.stack(list_of_crops[k:k+batch_size]).float().div_(255), imagenet_mean, imagenet_std).cuda(1), tensor(range(n_cells)))),axis=1).numpy() for k in range(0, n_cells, batch_size)])
                else:
                    predictions = np.array([])

                # Construct output mask and save information about objects
                output_mask1 = np.zeros(image_metas[4:6], dtype='bool')
                output_mask2 = np.zeros(image_metas[4:6], dtype='bool')
                for i in range(n_cells):
                    output_mask1[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = np.logical_or(output_mask1[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i])
                    if predictions[i] == cell_to_activate:
                        output_mask2[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = np.logical_or(output_mask2[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i])
                    global image_number
                    global celldata
                    global object_number
                    celldata[object_number] = [image_number, image_dir, object_number, i, cell_x_dim[i], cell_y_dim[i], cell_x_centroid[i], cell_y_centroid[i], cell_size[i], boundary_cell[i], predictions[i]]
                    object_number += 1
                output_mask = (output_mask1[padding[0][0]:image_metas[4]-padding[0][1],padding[1][0]:image_metas[5]-padding[1][1]].astype('uint8') + output_mask2[padding[0][0]:image_metas[4]-padding[0][1],padding[1][0]:image_metas[5]-padding[1][1]].astype('uint8'))
                
                # Save image information
                end_time2 = time.time()
                total_time = end_time2 - start_time
                global imagedata
                imagedata[image_number] = [image_number, image_dir, n_cells, n_cells-sum(predictions), segmentation_time, total_time]
                image_number += 1
                if image_number % save_every == 0:
                    #Save data as dataframes
                    image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated', 'SegmentationTime', 'TotalTime'])
                    cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Cell_Size', 'Boundary_Cell', 'Activated'])
                    image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
                    cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
                print('Total processing time in ms: ' + str(total_time*1000))
            
            return output_mask

if __name__ == "__main__":

    #Set global variables
    root_dir = "/home/srirampendyala/Projects/DSB_2018-master/"
    output_dir = "./barnyard/06-16-19_output"
    model_file = "deepretina_final.h5"
    model_path = os.path.join(root_dir, model_file)
    classification_model_path = '/home/srirampendyala/Projects/barnyard_experiment/preliminary_data/models_export_binned'
    cell_to_activate = 1 #3t3
    save_every = 500
    batch_size = 512
    expand_pixels = 2
    boundary_size = 5
    normalize_crops = False
    image_height = 768
    image_width = 870
    crop_size = 48
    celldata = {}
    image_number = 0
    imagedata = {}
    object_number = 0

    #Make necessary directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Initialize segmentation model
    inference_config = my_inference.BowlConfig()
    inference_config.display()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=root_dir)
    model.load_weights(model_path, by_name=True)

    #Initialize pytorch model
    from fastai.imports import *
    from fastai.vision import *
    torch.cuda.set_device(1)
    learn = load_learner(classification_model_path)
    imagenet_mean = tensor([0.485,0.456,0.406])
    imagenet_std = tensor([0.229,0.224,0.225])
    print('Pytorch is using GPU: ' + str(torch.cuda.current_device()))

    #Start rpyc server
    print('Ready to accept images!')
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(LinuxService, port = 18871).start()
