import sys

from skimage import img_as_float

sys.path.append('../DSB_2018-master')

import numpy as np
import skimage.io
import time
import os
import matplotlib.pyplot as plt
import random
from skimage.transform import downscale_local_mean
from skimage.draw import rectangle, rectangle_perimeter
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
import pandas as pd
import warnings
import argparse
from pathlib import Path
import tqdm
import logging
import pickle
import torch
import pprint
from decorators import pickler

#TODO - don't die on RuntimeError from pytorch
#TODO - Fix finalize method
#TODO - Every rpyc connect call launches a new ImageAnalysis class which is slow due to TF startup




# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable annoying tensorflow messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TimeIt(object):
    '''
    Quick and dirty object for code timing
    '''
    def __init__(self):
        self.start = time.time()

    def get_time(self):
        return time.time()-self.start

    def log(self, what):
        logger.debug(f"{what} took {self.get_time()} sec")


class VisualizeBase(object):
    def __init__(self,config,grid=(2,2),figsize=(12,12)):
        self.config = config
        self.grid = grid
        self.figsize = figsize
        self.setup_plots()

    def setup_plots(self):
        self.fig, self.ax = plt.subplots(*self.grid, sharex='col', sharey='row', figsize=self.figsize,num=1, clear=True)
        self.axiter = iter(self.ax.ravel())
        self.nax = next(self.axiter)

    def visualize_image(self, image, title,cmap=None):
        self.nax.set_title(title)
        self.nax.imshow(image,cmap=cmap)
        try:
            self.nax = next(self.axiter)
        except StopIteration:
            plt.draw()
            #plt.pause(0.5)
            plt.show()
            self.setup_plots()

class VisualizeClassifications(VisualizeBase):

    def visualize(self, predictions,crops,n_cells):
        if not self.config['visualize_classifications']: return
        d = {'edge': 1, 'noise': 2, 'puncta': 3, 'wt': 4}
        d = {y: x for x, y in d.items()}
        gridprod = np.product(self.grid)

        for i in range(0, len(predictions), gridprod):
            for k, predict in enumerate(predictions[i:i + gridprod]):
                title = f"{d[predict]}/{i + k}/{n_cells}"
                img = crops[k + i].numpy()[0]
                self.visualize_image(img,title,cmap='gray')

class VisualizeSegAndClass(VisualizeBase):
    def visualize_cell_boundaries(self, bboxes, image, title='',alpha=0.2,filter=None):
        if not self.config['visualize']: return
        nimage = img_as_float(image[..., 0], force_copy=True)
        nimage = np.stack([nimage] * 4, axis=2)
        nimage[:,:,3] = 1

        mask = np.zeros(nimage.shape)
        for bbox in bboxes:
            bbox = np.maximum(0, bbox)
            rr, cc = rectangle(bbox[[0, 1]], bbox[[2, 3]] - 1)
            mask[rr, cc, 0] = 1 # red
            mask[rr,cc,3] = alpha
        nimage = mask[...,0:3]*mask[...,3,None] + nimage[...,0:3]*(1-mask[...,3,None])
        self.visualize_image(nimage, title)

    def visualize_image(self, image, title,cmap=None):
        if not self.config['visualize']: return
        super().visualize_image(image,title,cmap=cmap)

class CellClassifier(object):
    model = None

    def __init__(self,config):
        self.config = config
        self.model = self.inference_config = None
        self.viz = None
        if self.config['visualize']:
            self.viz = VisualizeSegAndClass(self.config,grid=(2,2))
        if self.config['debug']:
            logger.setLevel(logging.DEBUG)


    def finalize(self):

        # Save data as dataframes
        imagedata = self.config['imagedata']
        celldata = self.config['celldata']
        image_dataframe = pd.DataFrame.from_dict(imagedata, orient='index',
                                                 columns=['ImageNumber', 'ImageDir', 'NCells', 'NCells_Activated',
                                                          'SegmentationTime', 'TotalTime'])
        cell_dataframe = pd.DataFrame.from_dict(celldata, orient='index',
                                                columns=['ImageNumber', 'ImageDir', 'ObjectNumber', 'CellNumber',
                                                         'Size_X', 'Size_Y', 'Centroid_X', 'Centroid_Y', 'Cell_Size',
                                                         'Boundary_Cell', 'Activated'])
        image_dataframe.to_csv(os.path.join(output_dir, 'images.csv'))
        cell_dataframe.to_csv(os.path.join(output_dir, 'cells.csv'))
        return

    def preprocess(self, image,normalize=True):
        if isinstance(image,str):
            image = skimage.io.imread(image)
        img = downscale_local_mean(image, self.config['binning'])

        img = (img.astype(np.single))[:,1:]

        if not normalize:
            return np.stack([img, img, img], axis=-1) #np_pixels_downscaled
        return CellClassifier.normalize_image(img)

    @staticmethod
    def normalize_image(img):

        # Normalize images
        percentile = 99.9
        high = np.percentile(img, percentile)
        low = np.percentile(img, 100 - percentile)

        img = np.minimum(high, img)
        img = np.maximum(low, img)

        img = np.minimum(1,(img - low) / (high - low))   # (0->1)

        # Make a RGBA-channel color image
        img_norm = np.stack([img, img, img], axis=-1)
        return img_norm

    def segment_nucleus(self,img_name,tf_gpu_fraction=None):
        '''
        Segment nuclei using the 3rd place segmenting algorithm from the Kaggle Data Science Bowl 2018
        :param img_name: File name of the microscope image to process
        :return: Dictionary: {'model_data':results, 'molded_image':molded_image, 'meta':meta}
                 where results is the dictionary returned by the DSB segmenting algorithm,
                 'molded_image' is the submitted image after processing, and meta is metadata.
        '''
        t0 = TimeIt()
        img = skimage.io.imread(img_name)
        full_path = str(Path(img_name).absolute())
        img = self.preprocess(img)
        if self.viz: self.viz.visualize_image(img, 'Dendra2 image (segment)')
        image_id = random.randint(0, 1<<31)



        from mrcnn import utils
        # Lazily initialize segmentation model
        import mrcnn.model_inference as modellib
        from mrcnn import my_inference
        if self.inference_config is None:
            self.inference_config = my_inference.BowlConfig()
            self.inference_config.display()
        inference_config = self.inference_config

        # Mold image
        molded_image, window, scale, padding, crop = utils.resize_image(img,
                                                                        min_dim=inference_config.IMAGE_MIN_DIM,
                                                                        min_scale=inference_config.IMAGE_MIN_SCALE,
                                                                        max_dim=inference_config.IMAGE_MAX_DIM,
                                                                        mode=inference_config.IMAGE_RESIZE_MODE)
        active_class_ids = [1, 1]
        meta = {
            'image_id' : random.randint(0, 1<<31),
            'source_image_path': full_path,
            'source_image_shape': img.shape,
            'molded_image_shape' : molded_image.shape,
            'window' : window,
            'scale' : scale,
            'active_class_ids' : active_class_ids,
            'padding' : padding,
            'crop' : crop
        }
        meta['preprocess_time'] = t0.get_time()

        # Load model if not already loaded
        if self.model is None:
            logger.info(f"Loading TF model in dir {self.config['root_dir']}")
            self.model = modellib.MaskRCNN(mode="inference", config=self.inference_config, model_dir=self.config['root_dir'])
            # Limit tf memory fraction if specified
            # Has to be located before loading weights or all sorts of chaos erupts
            if tf_gpu_fraction is not None:
                from keras.backend.tensorflow_backend import set_session, get_session
                import tensorflow as tf
                sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=tf_gpu_fraction),allow_soft_placement=True))
                set_session(sess)
            logger.info(f"Loading TF model weights from {self.config['model_path']}")
            self.model.load_weights(self.config['model_path'], by_name=True)
        logger.info(f"TF model loaded")
        model = self.model

        t0 = TimeIt()
        image_metas = modellib.compose_image_meta(image_id, img.shape, molded_image.shape, window,
                                                  scale, active_class_ids).astype(np.int32)
        # Model trained on 0->255 and we have 0->1 in our data, so need a multiply here
        results = model.detect_molded(np.expand_dims(molded_image*255., 0), np.expand_dims(image_metas, 0), verbose=0)
        meta['segment_time'] = t0.get_time()
        meta['ncells'] = 0
        if len(results) > 0:
            meta['ncells'] = len(results[0]['class_ids'])
        logger.debug(f"metadata from segmentation {meta}")
        if self.viz: self.viz.visualize_cell_boundaries(np.copy(results[0]['rois']).astype(np.int32), molded_image, title='Segmented cell boundaries')
        return {'model_data':results, 'molded_image':molded_image, 'meta':meta}


    @pickler
    def run(self, simg, cimg, tf_gpu_fraction=None):
        '''
        Segment and classify a multichannel image
        :param simg: Image to use for segmentation
        :param cimg: Image to use for classification
        :param tf_gpu_fraction: Fraction of GPU allocated for TensorFlow. If none, default to
        default TF memory allocation
        :return:
        '''
        segmented = self.segment_nucleus(simg,tf_gpu_fraction=tf_gpu_fraction)
        rval = self.classify_image(cimg,segmented)
        return rval

    def classify_image(self, img_name, segmented):
        '''
        Classify data from previously segmented nuclei generated from cell images
        :param img_name: file name of the image to classify
        :param segmented: Return value from the segmented_nucleus function
        :return: Dictionary: {'model_data':output_mask, 'meta':celldata} where the 'model_data' contains the
                            output mask generated by the classifier and 'meta' contains metadata from the
                            classification algorithm
        '''
        t0 = TimeIt()
        img = skimage.io.imread(img_name)
        full_path = str(Path(img_name).absolute())
        img = self.preprocess(img,normalize=False)
        image_id = random.randint(0, 1<<31)
        if self.viz: self.viz.visualize_image(self.normalize_image(img[...,0]), 'LMNA image (classify)')

        # Mold image
        from mrcnn import utils
        from mrcnn import my_inference
        if self.inference_config is None:
            self.inference_config = my_inference.BowlConfig()
        inference_config = self.inference_config
        molded_image, window, scale, padding, crop = utils.resize_image(img,
                                                                        min_dim=inference_config.IMAGE_MIN_DIM,
                                                                        min_scale=inference_config.IMAGE_MIN_SCALE,
                                                                        max_dim=inference_config.IMAGE_MAX_DIM,
                                                                        mode=inference_config.IMAGE_RESIZE_MODE)


        results = segmented['model_data']
        img_meta = segmented['meta']
        padding = img_meta['padding']
        n_cells = len(results[0]['class_ids'])
        logger.debug('Number of cells: ' + str(n_cells))
        # Compute crop dimensions, cell dimensions, centroids
        shape = np.array(img_meta['source_image_shape'])
        nshape = np.array(img_meta['molded_image_shape'])
        list_of_crops = []
        config = self.config

        # Get the segmented regions
        # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        cell_bb = np.copy(results[0]['rois']).astype(np.int32)

        # Expand the region by config['expand_pixels']
        crop_bb = np.zeros(cell_bb.shape,dtype=np.int32)
        crop_bb[:,0:2] = np.maximum(0, cell_bb[:,0:2] - config['expand_pixels'])
        crop_bb[:,2:4] = np.minimum(nshape[0:2]-1, cell_bb[:,2:4] + config['expand_pixels'])
        cell_dim = cell_bb[:,2:4] - cell_bb[:,0:2]
        cell_centroid = ((cell_bb[:,2:4] - cell_bb[:,0:2])/2.).astype(np.int32)
        # Ordering of cell_bb, crop_bb is ymin xmin ymax xmax
        # Ordering of padding is [(top, bottom), (left, right), (0, 0)]
        npadding = np.array(padding).reshape(6)[0:4][[1,2,0,3]]
        nboundary = cell_bb - npadding
        boundary_cell = np.any(np.logical_or(nboundary[:,0:2] <= config['boundary_size'],
                                             nboundary[:,2:4] >= (np.array((shape[0],shape[1])) - config['boundary_size'])),
                               axis=1)

        cell_size = np.zeros(n_cells, dtype='uint32')
        # Perform crops of images to put into classifier
        # roi['masks']: [H, W, N] instance binary masks where H,W dimensions of image, N cell num
        # Various bboxes: ymin,xmin,ymax,xmax,N
        crop_size = config['crop_size']
        pred_to_index = {}
        for i in range(n_cells):
            # Mask the image using segmentation data
            bbox_slice = [slice(*cell_bb[i,[0,2]]), slice(*cell_bb[i,[1,3]])]
            mask = results[0]['masks'][i][bbox_slice]
            cell_size[i] = np.sum(mask,axis=(0,1))
            crop = molded_image[bbox_slice + [0]]
            if cell_size[i] <= config['cell_size_minimum'] or boundary_cell[i]:
                continue
            crop = crop * mask


            # Center cell image in config['crop_size]x config['crop_size'] region
            cshapes = np.array(crop.shape)
            crop_min = (crop_size - cshapes)//2
            padsmin = np.maximum(0,crop_min)
            crop_max = crop_size - padsmin - cshapes
            padsmax = np.maximum(0,crop_max)
            # Pad if image is too small
            crop = np.pad(crop,((padsmin[0],padsmax[0]),(padsmin[1],padsmax[1])))

            # Crop if image is too large
            crop_max = crop_size - crop_min - cshapes
            crop_min = np.maximum(0,-crop_min)
            crop_max = np.maximum(0,-crop_max)
            crop = skimage.util.crop(crop,((crop_min[0],crop_max[0]),(crop_min[1],crop_max[1])))


            # Keep track of predict index to original index
            pred_to_index[len(list_of_crops)] = i
            # Normalize cell, then convert cell to tensor and move to gpu
            crop = self.normalize_image(crop)
            list_of_crops.append(torch.from_numpy(np.moveaxis(crop, -1, 0)))

        preprocess_time = t0.get_time()

        # Perform classifier predictions on gpu
        # Initialize pytorch model
        logger.debug(f"Loading PyTorch classification model from {config['classification_model_path']}")
        from fastai.basic_data import DatasetType
        from fastai.basic_train import load_learner
        from fastai.torch_core import tensor
        from fastai.vision.data import normalize
        t0 = TimeIt()
        learn = load_learner(config['classification_model_path'],config['classification_model_file'])
        logger.debug(f"PyTorch model loaded")
        imagenet_mean = tensor([0.485, 0.456, 0.406])
        imagenet_std = tensor([0.229, 0.224, 0.225])
        if n_cells > 0:
            predictions = []
            for k in range(0, n_cells, config['batch_size']):
                predict = np.argmax(
                    learn.pred_batch(ds_type=DatasetType.Test,
                                     batch=(normalize(
                                         torch.stack(list_of_crops[k:k + config['batch_size']]).float(),
                                         imagenet_mean,
                                         imagenet_std).cuda(),
                                            tensor(range(n_cells)))), axis=1).numpy()+1
                predictions.append(predict)
            if self.config['visualize_classifications']:
                vc = VisualizeClassifications(self.config,grid=(4,4)).visualize(predictions[0],list_of_crops,n_cells)
            predictions = np.concatenate(predictions)
            #predictions = np.random.randint(4, size=n_cells)+1
        else:
            predictions = np.array([])
        classify_time = t0.get_time()

        # Construct output mask and save information about objects
        output_mask = np.zeros((nshape[0],nshape[1]), dtype=np.int8)
        image_number = img_meta['image_id']
        celldata = {}
        for j,i in pred_to_index.items():
            bbox_slice = [slice(*cell_bb[i,[0,2]]), slice(*cell_bb[i,[1,3]])]
            seg_mask = results[0]['masks'][i][bbox_slice] * predictions[j]
            output_mask[bbox_slice] = np.bitwise_or(output_mask[bbox_slice],seg_mask)
            celldata[i] = dict(
                zip(
                    ('image_id','source_image_path','dims','centroid','cell_size',
                     'boundary_cell', 'activated','preprocess_time','classify_time'),
                    (image_number, img_meta['source_image_path'], cell_dim[i],
                                       cell_centroid[i], cell_size[i], boundary_cell[i],
                                       predictions[j],classify_time,preprocess_time)
                ))
        output_mask = output_mask[padding[0][0]:nshape[0] - padding[0][1], padding[1][0]:nshape[1] - padding[1][1]]
        if not output_mask.flags.c_contiguous:
            output_mask = output_mask.copy(order='C')
        if self.viz: self.viz.visualize_image(output_mask, 'Predictions')
        logger.debug(f"metadata from classification {celldata}")
        return {'model_data':output_mask, 'meta':celldata}


if __name__ == "__main__":
    def validate_input(val):
        if not Path(val).is_dir():
            raise argparse.ArgumentTypeError(f"{val} is not a directory or isn't readable")
        return val

    def validate_output(val):
        try:
            Path(val).mkdir(exist_ok=True)
        except:
            raise argparse.ArgumentTypeError(f"{val} cannot be created")
        return val

    def validate_range(val):
        try:
            val = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{val} is not a floating point number)")
        if val < 0. or val > 1.:
            raise argparse.ArgumentTypeError(f"Invalid range: {val} must be > 0. and <= 1.)")
        return val


    parser = argparse.ArgumentParser(description='Segment and/or classify an image')
    parser.add_argument('action', help="Operation to perform: segment | classify | full (both segment and classify)", choices=['segment','classify',"full"])
    parser.add_argument('--input_dir', help="Input directory of images", type=validate_input,required=True)
    parser.add_argument('--seg_dir',help="Directory for generated segmentation data.", type=validate_output)
    parser.add_argument('--seg_pattern', help="Glob expression for files in input directory that are used for segmentation", default='*')
    parser.add_argument('--classify_pattern', help="Glob expression for files in input directory that are used for classification", default='*')
    parser.add_argument('--max_images', help="Maximum number of images to process", type=int, default=-1)
    parser.add_argument('--tf_gpu_fraction', help="Limit tensorflow memory allocation to this fraction [0-1)", type=validate_range, default=None)
    parser.add_argument('--debug', help="Print debugging statements", action="store_true")
    parser.add_argument('--visualize', help="Visualize results", action="store_true")
    parser.add_argument('--visualize_classifications', help="Visualize classification results", action="store_true")
    args = parser.parse_args()
    if args.action == 'classify' and args.seg_dir is None:
        parser.error("--seg_dir argument is missing")
    if args.action == 'segment' and args.seg_dir is None:
        logger.warning("No --seg_dir argument so no data will be written")


    logger.debug('Pytorch is using GPU: ' + str(torch.cuda.current_device()))

    #Set configuration
    root_dir = "../DSB_2018-master/"
    model_file = "deepretina_final.h5"
    config = {
        'root_dir' : root_dir,
        'model_file': model_file,
        'model_path':os.path.join(root_dir, model_file),
        'classification_model_path':'./models/imageset_divided/2x2_binning', #use hyeon-jin's resnet34 model trained on 2x2 binned crops with multiple classes,
        'classification_model_file':'export.pkl',
         #'classification_model_path':'../classifier-images/imageset_divided',
         #'classification_model_file' : 'model_test_64.20200120_1303',
        'save_every':500,
        'cell_size_minimum':0,
        'batch_size':512,
        'expand_pixels':2,
        'boundary_size':3,
        'crop_size':64,
        'visualize':args.visualize,
        'visualize_classifications' : args.visualize_classifications,
        'binning': (2,2),
        'debug': args.debug
    }
    logger.info("Configuration:")
    logger.info(pprint.pformat(config))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        mon = CellClassifier(config)
        if args.action == 'segment':
            files = sorted(list(Path(args.input_dir).glob(args.seg_pattern)))

            if args.max_images > 0: files = files[0:args.max_images]
            outfiles = None
            if args.seg_dir:
                outfiles = [(Path(args.seg_dir) / p.parts[-1]).with_suffix('.pkl') for p in files]
            files = tqdm.tqdm(files)
            for i,f in enumerate(files):
                files.set_description(f"Segmenting {f}")
                results = mon.segment_nucleus(f,tf_gpu_fraction=args.tf_gpu_fraction)
                if outfiles is not None:
                    with open(outfiles[i],"wb") as f:
                       pickle.dump(results,f)
        elif args.action == 'classify':
            files = sorted(list(Path(args.input_dir).glob(args.classify_pattern)))
            if args.max_images > 0: files = files[0:args.max_images]
            seg_files = sorted(list(Path(args.seg_dir).glob('*.pkl')))
            if args.max_images > 0: seg_files = seg_files[0:args.max_images]
            assert len(seg_files) == len(files)
            files = tqdm.tqdm(files)
            for i,file in enumerate(files):
                with (open(seg_files[i],"rb")) as f:
                    files.set_description(f"Classifying {file} with segmentation data from {seg_files[i]}")
                    segmented = pickle.load(f)
                    results = mon.classify_image(file, segmented)
        elif args.action == 'full':
            cfiles = sorted(list(Path(args.input_dir).glob(args.classify_pattern)))
            sfiles = sorted(list(Path(args.input_dir).glob(args.seg_pattern)))
            if args.max_images > 0: cfiles = cfiles[0:args.max_images]
            if args.max_images > 0: sfiles = sfiles[0:args.max_images]
            assert len(cfiles) == len(sfiles)
            sfiles = tqdm.tqdm(sfiles)
            for i,sfile in enumerate(sfiles):
                results = mon.run(sfile, cfiles[i], tf_gpu_fraction=args.tf_gpu_fraction)
                # with (open("/tmp/bad.pkl","wb")) as f:
                #     pickle.dump(results,f)
        else:
            # Should never reach here
            raise Exception(f"Invalid analysis type {args.action}")



