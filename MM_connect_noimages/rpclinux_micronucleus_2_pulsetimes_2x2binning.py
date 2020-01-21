import rpyc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1,0'

classification_model_path = '/home/srirampendyala/Projects/RPE-micronucleus-training/data_unet_large_total/models_exported/resnet18_96x96crops_noclass/' #use micronucleus class-independent resnet18 trained on 96x96 2x2 binned normalized crops with sobel
#Initialize pytorch model
from fastai.imports import *
from fastai.vision import *
torch.cuda.set_device(1)
learn = load_learner(classification_model_path)
imagenet_mean = tensor([0.485,0.456,0.406])
imagenet_std = tensor([0.229,0.224,0.225])
print('Pytorch is using GPU: ' + str(torch.cuda.current_device()))

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

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import regionprops, label
from skimage.transform import rescale

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
            cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Cell_Size', 'Boundary_Cell', 'N_Micronuclei', 'Micronuclei_Called'])
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
            #import image
            img = np.frombuffer(base64.decodebytes(img), np.uint8).reshape((image_height,image_width,2))
            np.save(os.path.join(image_dir,'image.npy'), img)

            #analyze
            output_mask = self.analyze_image(image_name, img, image_dir)

            #save output mask
            skimage.io.imsave(os.path.join(image_dir, 'mask.png'), output_mask)

            #encode base64 and output
            if not output_mask.flags.c_contiguous:
                output_mask = output_mask.copy(order='C')
            return base64.b64encode(output_mask)

            #t.Thread(target=self.save_files, args=(image_name, img, output_mask)).start()
            #mp.Process(target=self.save_files, args=(image_name, img, output_mask)).start()

        def normalize_image(self, img):
    
            if len(img.shape) == 2:
        
                percentile = 99.9
                high = np.percentile(img, percentile)
                low = np.percentile(img, 100-percentile)

                img = np.minimum(high, img)
                img = np.maximum(low, img)

                img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
                img = skimage.img_as_ubyte(img)

                # make a RGB-channel color image
                img_norm = np.stack([img, img, img], axis=-1)
        
            else:
                img_norm = np.zeros(img.shape, dtype='uint8')
                for ix in range(img.shape[2]):
            
                    img_channel =  img[:,:,ix]
                    percentile = 99.9
                    high = np.percentile(img_channel, percentile)
                    low = np.percentile(img_channel, 100-percentile)

                    img_channel = np.minimum(high, img_channel)
                    img_channel = np.maximum(low, img_channel)

                    img_channel = (img_channel - low) / (high - low) # gives float64, thus cast to 8 bit later
                    img_norm[:,:,ix] = skimage.img_as_ubyte(img_channel)
            
            return img_norm

        def analyze_image(self, image_name, image, image_dir):

            # Catch warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Mold image
                start_time = time.time()
                
                #create 3 channel image for segmentation
                image_3channel = np.insert(image, 2, 0, axis=2)
                img_seg = self.normalize_image(image_3channel)
                
                #define miRFP and Dendra image
                miRFP_image = image[:,:,0]
                Dendra_image = image[:,:,1]
                
                molded_image, window, scale, padding, crop = utils.resize_image(img_seg, min_dim=inference_config.IMAGE_MIN_DIM, min_scale=inference_config.IMAGE_MIN_SCALE, max_dim=inference_config.IMAGE_MAX_DIM, mode=inference_config.IMAGE_RESIZE_MODE)
                active_class_ids = [1,1]
                image_metas = modellib.compose_image_meta(0, (image_height,image_width,3), molded_image.shape, window, scale, active_class_ids)
                
                #define normalized miRFP and Dendra image, and pad them
                miRFP_image_norm = img_seg[:,:,0]
                miRFP_image_pad = np.pad(miRFP_image, padding[0:2], mode='constant')
                miRFP_image_norm_pad = np.pad(miRFP_image_norm, padding[0:2], mode='constant')

                Dendra_image_norm = img_seg[:,:,1]
                Dendra_image_pad = np.pad(Dendra_image, padding[0:2], mode='constant')
                Dendra_image_norm_pad = np.pad(Dendra_image_norm, padding[0:2], mode='constant')

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
                cell_y_dim = cell_y_max - cell_y_min
                cell_y_centroid = np.int_((cell_y_min + cell_y_max)/2)
                crop_y_min = np.maximum(0,cell_y_centroid - crop_size//2)
                crop_y_max = np.minimum(image_metas[4],cell_y_centroid + crop_size//2)
                crop_y_dim = crop_y_max - crop_y_min

                cell_x_min = results[0]['rois'][:,1]
                cell_x_max = results[0]['rois'][:,3]
                cell_x_centroid = np.int_((cell_x_min + cell_x_max)/2)
                crop_x_min = np.maximum(0, cell_x_centroid - crop_size//2)
                crop_x_max = np.minimum(image_metas[5], cell_x_centroid + crop_size//2)
                cell_x_dim = cell_x_max - cell_x_min
                crop_x_dim = crop_x_max - crop_x_min
                cell_centroids = np.stack([cell_y_centroid,cell_x_centroid],axis=-1)

                cell_size = np.zeros(n_cells, dtype='uint32')
                boundary_cell = (((cell_x_min - padding[1][1]) <= boundary_size) | ((cell_x_max - padding[1][1]) >= image_metas[2] - boundary_size)) | (((cell_y_min - padding[0][1]) <= boundary_size) | ((cell_y_max - padding[0][1]) >= image_metas[1] - boundary_size))

                mean_nuclear_miRFP_intensity = np.zeros(n_cells, dtype='float32')
                mean_nuclear_Dendra_intensity = np.zeros(n_cells, dtype='float32')
                n_micronuclei = np.zeros(n_cells, dtype='uint32')

                # Perform crops of images to put into classifier
                for i in range(n_cells):
                    
                    # Crop cell in form of mask into crop_size x crop_size
                    cell_size[i] = sum(sum(results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i]))
                    crop = np.stack([miRFP_image_pad[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], Dendra_image_pad[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]]], axis=-1)
                    crop = np.insert(crop, 2, 0, axis=2)
                    mask = results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i]

                    Dendra_crop = crop[:,:,1] * mask
                    miRFP_crop = crop[:,:,0] * mask

                    mean_nuclear_Dendra_intensity[i] = np.true_divide(Dendra_crop.sum(), (Dendra_crop!=0).sum())
                    mean_nuclear_miRFP_intensity[i] = np.true_divide(miRFP_crop.sum(),(miRFP_crop!=0).sum())

                    #normalize crop
                    crop = self.normalize_image(crop)

                    #add blue channel
                    crop_norm_bw = rgb2gray(crop)
                    crop[:,:,2] = self.normalize_image(sobel(crop_norm_bw))[:,:,0]

                    y_pad = (crop_size - crop_y_dim[i])//2
                    y_odd = crop_y_dim[i] % 2
                    x_pad = (crop_size - crop_x_dim[i])//2
                    x_odd = crop_x_dim[i] % 2

                    crop = np.pad(crop, ((max(0,y_pad),max(0,y_pad + y_odd)), (max(0,x_pad),max(0,x_pad + x_odd)), (0,0)), mode='constant')[max(0,-y_pad):max(0,-y_pad)+crop_size, max(0,-x_pad):max(0,-x_pad)+crop_size,:]

                    #rescale crop by 2x
                    crop = rescale(crop, (2,2,1), anti_aliasing=False)
                    
                    # Normalize cell, then convert cell to tensor and move to gpu 
                    list_of_crops.append(torch.from_numpy(np.moveaxis(crop, -1, 0)))

                    # Save crops
                    np.save(os.path.join(image_dir,'cell_images',str(i)+'.npy'), crop)
            
                # Perform UNet predictions on gpu
                if n_cells > 0:
                    predictions = np.concatenate([np.argmax(learn.pred_batch(ds_type=DatasetType.Test, batch=(normalize(torch.stack(list_of_crops[k:k+batch_size]).float(), imagenet_mean, imagenet_std).cuda(1), tensor(range(n_cells)))),axis=1).numpy() for k in range(0, n_cells, batch_size)])
                    micronuclei_calls = np.zeros(n_cells, dtype='bool')
                else:
                    predictions = np.array([])
                    micronuclei_calls = np.array([])
        
                #combine all nuclear predictions
                nuclear_masks = np.zeros(image_metas[4:6], dtype='bool')
                for i in range(n_cells):
                    nuclear_masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = nuclear_masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = np.logical_or(nuclear_masks[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i])
                
                #iterate over cells
                for i in range(n_cells):
                    #segment micronuclei from UNet predictions
                    micronuclei_regionprops = regionprops(label((predictions[i,:,:]==1)*1))
                    n_micronuclei[i] = len(micronuclei_regionprops)
        
                    #localize micronuclei in the parent image
                    micronuclei_y_centroid = [cell_y_centroid[i]]*n_micronuclei[i]
                    micronuclei_x_centroid = [cell_x_centroid[i]]*n_micronuclei[i]
        
                    micronuclei_y_min = [0]*n_micronuclei[i]
                    micronuclei_y_max = [0]*n_micronuclei[i]
                    micronuclei_x_min = [0]*n_micronuclei[i]
                    micronuclei_x_max = [0]*n_micronuclei[i]
        
                    check_inside_nuclei = np.zeros(n_micronuclei[i], dtype='bool')
                    for j in range(n_micronuclei[i]):
                        micronuclei_y_centroid[j] += micronuclei_regionprops[j].centroid[0]/2 - crop_size
                        micronuclei_x_centroid[j] += micronuclei_regionprops[j].centroid[1]/2 - crop_size
            
                        #verify that they are not inside segmented nuclei
                        check_inside_nuclei[j] = nuclear_masks[np.int_(micronuclei_y_centroid[j]), np.int_(micronuclei_x_centroid[j])]
            
                    #find closest nucleus and associate the micronucleus to that nucleus
                    dist_micronuclei_matrix = scipy.spatial.distance.cdist(np.stack([micronuclei_y_centroid, micronuclei_x_centroid],axis=-1), cell_centroids)
                    nuclear_indices = np.argmin(dist_micronuclei_matrix, axis=-1)
                    
                    #compute distance from micronucleus to nucleus
                    dist_micronuclei = np.array([dist_micronuclei_matrix[i,j] for (i,j) in enumerate(nuclear_indices)])
                    
                    #set all micronuclei calls to 1 for associated nuclei, as long as the micronucleus is not inside the nuclear segmentation and as long as the micronucleus is less than the minimum distance
                    micronuclei_calls[nuclear_indices[(~check_inside_nuclei) & (dist_micronuclei < max_dist_micronuclei)]] = True
                    
                #construct output mask and save information about objects
                output_mask1 = np.zeros(image_metas[4:6], dtype='bool')
                output_mask2 = np.zeros(image_metas[4:6], dtype='bool')
                n_activated_cells = 0
                for i in range(n_cells):
                    #construct output mask
                    if ((cell_size[i] >= cell_size_cutoff) & (boundary_cell[i] == 0)):
                        output_mask1[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = np.logical_or(output_mask1[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i])
                    if ((micronuclei_calls[i]) & (cell_size[i] >= cell_size_cutoff) & (boundary_cell[i] == 0)):
                        output_mask2[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]] = np.logical_or(output_mask2[crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i]], results[0]['masks'][crop_y_min[i]:crop_y_max[i],crop_x_min[i]:crop_x_max[i],i])
                        n_activated_cells += 1
                    #save information about objects
                    global image_number
                    global celldata
                    global object_number
                    celldata[object_number] = [image_number, image_dir, object_number, i, cell_x_dim[i], cell_y_dim[i], cell_x_centroid[i], cell_y_centroid[i], cell_size[i], boundary_cell[i], n_micronuclei[i], micronuclei_calls[i]]
                    object_number += 1   
                output_mask = (output_mask1[padding[0][0]:image_metas[4]-padding[0][1],padding[1][0]:image_metas[5]-padding[1][1]].astype('uint8') + output_mask2[padding[0][0]:image_metas[4]-padding[0][1],padding[1][0]:image_metas[5]-padding[1][1]].astype('uint8'))

                # Save image information
                end_time2 = time.time()
                total_time = end_time2 - start_time
                global imagedata
                imagedata[image_number] = [image_number, image_dir, n_cells, n_activated_cells, segmentation_time, total_time]
                image_number += 1
                if image_number % save_every == 0:
                    #Save data as dataframes
                    image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated', 'SegmentationTime', 'TotalTime'])
                    cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Cell_Size', 'Boundary_Cell', 'N_Micronuclei', 'Micronuclei_Called'])
                    image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
                    cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
                print('Total processing time in ms: ' + str(total_time*1000))
                print('Number of activated cells: ' + str(n_activated_cells))
            
            return output_mask

if __name__ == "__main__":

    #Set global variables
    root_dir = "/home/srirampendyala/Projects/DSB_2018-master/"
    output_dir = "./micronucleus/test_12_6_19/"
    model_file = "deepretina_final.h5"
    model_path = os.path.join(root_dir, model_file)   

    save_every = 300
    cell_size_cutoff = 0
    batch_size = 64
    expand_pixels = 2
    boundary_size = 3
    image_height = 768
    image_width = 869
    crop_size = 48
    max_dist_micronuclei = 40
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

    #Start rpyc server
    print('Ready to accept images!')
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(LinuxService, port = 18871).start()
