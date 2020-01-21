import rpyc
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import threading as t
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
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
import PIL.Image as Image
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
            #finalize the CNN
            #Save data as dataframes
            image_dataframe = pd.DataFrame.from_dict(imagedata,orient='index',columns=['ImageNumber', 'ImageDir', 'NCells', 'SegmentationTime', 'TotalTime'])
            cell_dataframe = pd.DataFrame.from_dict(celldata,orient='index',columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber', 'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Saved', 'Activated'])
            image_dataframe.to_csv(os.path.join(output_dir,'images.csv'))
            cell_dataframe.to_csv(os.path.join(output_dir,'cells.csv'))
            return

        def exposed_run_pipeline_on_image(self, image_name, img):
            image_name = os.path.splitext(image_name)[0]
            image_dir = os.path.join(output_dir, image_name)
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
                os.mkdir(os.path.join(image_dir, 'cell_images'))
                os.mkdir(os.path.join(image_dir, 'masks'))
            #import image
            img = np.frombuffer(base64.decodebytes(img), np.uint16).reshape((image_height,image_width))

            #normalize and save
            img_norm = self.normalize(img)
            #skimage.io.imsave(os.path.join(working_dir,'output',image_name + '_Norm.png'), img_norm)

            #analyze
            output_mask = self.analyze_image(image_name, img_norm, image_dir)

            #encode base64 and output
            if not output_mask.flags.c_contiguous:
                output_mask = output_mask.copy(order='C')
            return base64.b64encode(output_mask)

            #t.Thread(target=self.save_files, args=(image_name, img, output_mask)).start()
            #mp.Process(target=self.save_files, args=(image_name, img, output_mask)).start()

        def analyze_image(self, image_name, image, image_dir):

            # Catch warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Mold image
                molded_image, window, scale, padding, crop = utils.resize_image(image, min_dim=inference_config.IMAGE_MIN_DIM, min_scale=inference_config.IMAGE_MIN_SCALE, max_dim=inference_config.IMAGE_MAX_DIM, mode=inference_config.IMAGE_RESIZE_MODE);
                active_class_ids = [1,1];
                image_metas = modellib.compose_image_meta(0, (image_height,image_width,3), molded_image.shape, window, scale, active_class_ids);

                # Run object detection
                start_time = time.time();
                results = model.detect_molded(np.expand_dims(molded_image, 0), np.expand_dims(image_metas, 0), verbose=1);
                end_time1 = time.time();
                segmentation_time = end_time1-start_time;
                n_masks = results[0]['masks'].shape[2];

                # Save cell-level parameters
                cell_y_min = results[0]['rois'][:,0]
                cell_y_max = results[0]['rois'][:,2]
                cell_x_min = results[0]['rois'][:,1]
                cell_x_max = results[0]['rois'][:,3]
                cell_y_dim = cell_y_max - cell_y_min;
                cell_x_dim = cell_x_max - cell_x_min;
                cell_y_centroid = np.int_((cell_y_min + cell_y_max)/2);
                cell_x_centroid = np.int_((cell_x_min + cell_x_max)/2);

                # Analyze each cell and generate output mask
                output_mask = np.zeros(image_metas[4:6], dtype='bool');
                for j in range(n_masks):
                    mask = results[0]['masks'][:,:,j];
                    activate = self.analyze_cell(mask, molded_image, image_metas[4:6], j, image_dir, cell_y_dim[j], cell_x_dim[j], cell_y_centroid[j], cell_x_centroid[j]);
                    if activate:
                        output_mask[cell_y_min[j]:cell_y_max[j],cell_x_min[j]:cell_x_max[j]] = np.logical_or(output_mask[cell_y_min[j]:cell_y_max[j],cell_x_min[j]:cell_x_max[j]], mask[cell_y_min[j]:cell_y_max[j],cell_x_min[j]:cell_x_max[j]]);
                output_mask = output_mask[padding[0][0]:image_metas[4]-padding[0][1],padding[1][0]:image_metas[5]-padding[1][1]]
                #skimage.io.imsave(os.path.join(working_dir,'output',image_name + '_Binary.png'),(255*output_mask.astype(np.uint8)))

                # Save image information
                end_time2 = time.time();
                total_time = end_time2 - start_time;
                global imagedata
                global image_number
                imagedata[image_number] = [image_number, image_dir, n_masks, segmentation_time, total_time];
                image_number += 1;

                # Print information
                print(str(n_masks) + ' cells');
                print(str(segmentation_time) + ' s to segment');
                print(str(total_time) + ' s total');
                if n_masks > 0:
                    print(str(segmentation_time/n_masks*1000) + ' ms per cell to segment')
                    print(str(total_time/n_masks*1000) + ' ms per cell total')
            return output_mask
        
        def analyze_cell(self, mask, image, image_size, cell_number, image_dir, cell_y_dim, cell_x_dim, cell_y_centroid, cell_x_centroid):
            # Analyze each cell individually
            activate=True;
            saved=False;
            if max(cell_y_dim,cell_x_dim) < cell_image_size:
                x_crop = np.int_([cell_x_centroid-cell_image_size/2,cell_x_centroid+cell_image_size/2]);
                y_crop = np.int_([cell_y_centroid-cell_image_size/2,cell_y_centroid+cell_image_size/2]);
                cell_image = self.crop_cell_nobackground(image[:,:,0], image_size, mask, x_crop, y_crop);
                #skimage.io.imsave(os.path.join(image_dir, 'cell_images', str(cell_number) + '.png'), cell_image); #save cell
                saved=True;
            global celldata
            global object_number
            celldata[object_number] = [image_number, image_dir, object_number, cell_number, cell_x_dim, cell_y_dim, cell_x_centroid, cell_y_centroid, saved, activate];
            object_number += 1;
            return activate;

        def normalize(self, img):
            #normalize images
            percentile = 99.9;
            high = np.percentile(img, percentile);
            low = np.percentile(img, 100-percentile);

            img = np.minimum(high, img);
            img = np.maximum(low, img);

            img = (img - low) / (high - low); # gives float64, thus cast to 8 bit later
            img = skimage.img_as_ubyte(img);

            # make a RGBA-channel color image
            img_norm = np.stack([img, img, img], axis=-1);
            return img_norm;

        def crop_cell_nobackground(self, image, image_size, mask, x_coords, y_coords):
            #Crop only the cell
            #Determine amount needed to pad arrays
            pad_sizes = ((max(0,0-y_coords[0]),max(0,y_coords[1]-image_size[0])),(max(0,0-x_coords[0]),max(0,x_coords[1]-image_size[1])));
            x_coords = x_coords + pad_sizes[1][0];
            y_coords = y_coords + pad_sizes[0][0];

            #Pad arrays by the necessary amount
            mask = np.pad(mask, pad_sizes, 'constant');
            image = np.pad(image, pad_sizes, 'constant');
    
            #Perform crop
            return np.multiply(mask[y_coords[0]:y_coords[1],x_coords[0]:x_coords[1]],image[y_coords[0]:y_coords[1],x_coords[0]:x_coords[1]]);

        def crop_cell_incontext(self, image, image_size, x_coords, y_coords):
            #Crop the cell in context
            #Determine amount needed to pad arrays
            pad_sizes = [(max(0,0-y_coords[0]),max(0,y_coords[1]-image_size[0])),(max(0,0-x_coords[0]),max(0,x_coords[1]-image_size[1]))];

            #Pad arrays by the necessary amount
            image = np.pad(image, pad_sizes, 'constant');
    
            #Perform crop
            return image[y_coords[0]:y_coords[1],x_coords[0]:x_coords[1]];

        def save_files(self, image_name, img, output_mask):
            open("tif_image"+str(image_name)+".tif", "wb").write(img)
            open("png_mask"+str(image_name)+".png", "wb").write(output_mask)

if __name__ == "__main__":

    #Set global variables
    root_dir = "/home/srirampendyala/Projects/DSB_2018-master/"
    working_dir = "./working/"
    output_dir = "./test/"
    model_file = "deepretina_final.h5"
    model_path = os.path.join(root_dir, model_file)
    image_height = 1536
    image_width = 1740
    cell_image_size = 128
    celldata = {}
    image_number = 0
    imagedata = {}
    object_number = 0

    #Make necessary directories
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Initialize model
    inference_config = my_inference.BowlConfig()
    inference_config.display()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=root_dir)
    model.load_weights(model_path, by_name=True)

    #Start rpyc server
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(LinuxService, port = 18871).start()
