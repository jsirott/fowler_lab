import re
import sys

from skimage import img_as_float

sys.path.append('../DSB_2018-master')

import numpy as np
#TODO- write metadata file
#TODO - figure out how to get class names from learner
#TODO -- make sure that fast.ai learner was trained on image that is same as that specified by config file
#TODO - don't die on RuntimeError from pytorch
#TODO - Fix finalize method
#TODO - Every rpyc connect call launches a new ImageAnalysis class which is slow due to TF startup
#TODO - Following fails with skimage. Look at this later because speedups are remarkable with cupy
#TODO - Very sensitive to binning/crop size. Some images from different datasets have different data sizes

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
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
from nucleus import nucleus
import matplotlib.patches as patches
from mrcnn import my_inference
from scipy.sparse.bsr import bsr_matrix
import numpy.ma as ma

from decorators import pickler

pd.set_option('max_colwidth',500)
pd.set_option('max_columns', 100)
pd.set_option('display.width',1000)
pd.set_option('max_rows',1000)






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
    def __init__(self,classifier,grid=(2,2),figsize=(12,12)):
        self.config = classifier
        self.grid = grid
        self.figsize = figsize
        self.setup_plots()
        self.texts = []

    def set_title(self, name):
        self.fig.suptitle(name)

    def setup_plots(self):
        self.fig, self.ax = plt.subplots(*self.grid, figsize=self.figsize,num=1, clear=True)
        self.fig.set_tight_layout(True)
        self.axiter = iter(self.ax.ravel())
        self.nax = next(self.axiter)

    def visualize_image(self, image, title,cmap=None):
        self.nax.set_title(title)
        self.nax.imshow(image,cmap=cmap)
        for args,kwargs in self.texts:
            self.nax.text(*args,**kwargs)
        self.texts = []
        try:
            self.nax = next(self.axiter)
        except StopIteration:
            plt.draw()
            plt.waitforbuttonpress(60 * 30)
            self.setup_plots()

    def text(self,*args, **kwargs):
        self.texts.append((args,kwargs))


class VisualizeClassifications(VisualizeBase):

    def visualize(self,predictions,crops,n_cells):
        if not self.config['visualize_classifications']: return
        #d = {'edge': 1, 'noise': 2, 'puncta': 3, 'wt': 4}
        d = {'discard': 1, 'puncta': 2, 'wt': 3}
        d = {y: x for x, y in d.items()}
        gridprod = np.product(self.grid)

        for i in range(0, len(predictions), gridprod):
            for k, predict in enumerate(predictions[i:i + gridprod]):
                title = f"{d[np.argmax(predict)+1]}/{np.max(predict):.2}/{i + k}/{n_cells}"
                img = crops[k + i][0]
                self.visualize_image(img,title,cmap='gray')


class VisualizeSegAndClass(VisualizeBase):

    def visualize_cell_boundaries(self, results, image, title='',alpha=0.2):
        bboxes = results[0]['rois'].astype(np.int32)
        masks = results[0]['masks']
        scores = results[0]['scores']

        nimage = img_as_float(image[..., 0], force_copy=True)
        nimage = np.stack([nimage] * 4, axis=2)
        nimage[:,:,3] = 1

        mask = np.zeros(nimage.shape)
        for i,bbox in enumerate(bboxes):
            bbox = np.maximum(0, bbox)
            # Bounding box
            rr, cc = rectangle_perimeter(bbox[[0, 1]], bbox[[2, 3]] - 2)
            mask[rr, cc, i%3] = 1 # Cycle through RGB
            mask[rr,cc,3] = 1

            # Nucleus mask
            mcoords = np.where(masks[i])
            mask[mcoords[0], mcoords[1], i%3] = 1
            mask[mcoords[0], mcoords[1] ,3] = alpha
            self.text(*bbox[[1,0]],f"{scores[i]:.3f}",color='w', size=7, backgroundcolor="none")
        nimage = mask[...,0:3]*mask[...,3,None] + nimage[...,0:3]*(1-mask[...,3,None])
        self.visualize_image(nimage, title)

class MaskRCNNInitializer(object):
    def __init__(self,classifier):
        self.classifier = classifier
        self.model = None

    def __call__(self, *args, **kwargs):
        if self.model is None:
            from mrcnn import utils
            import mrcnn.model_inference as modellib
            logger.info(f"Loading TF model in dir {self.classifier.config['root_dir']}")
            self.model = modellib.MaskRCNN(mode="inference", config=self.classifier.inference_config, model_dir=self.classifier.config['root_dir'])
            # Limit tf memory fraction if specified
            # Has to be located before loading weights or all sorts of chaos erupts
            tf_gpu_fraction = self.classifier.config['tf_gpu_fraction']
            if tf_gpu_fraction is not None:
                from keras.backend.tensorflow_backend import set_session, get_session
                import tensorflow as tf
                sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=tf_gpu_fraction),allow_soft_placement=True))
                set_session(sess)
            logger.info(f"Loading TF model weights from {self.classifier.config['model_path']}")
            self.model.load_weights(self.classifier.config['model_path'], by_name=True)
            logger.info(f"TF model loaded")
        return self.model



class CellClassifier(object):
    model = None

    def __init__(self,config):
        self.config = config
        self.inference_config = my_inference.BowlConfig()
        self.inference_config.display()
        self.model = MaskRCNNInitializer(self)
        self.viz = None
        self.vizseg = None
        if self.config['visualize_classifications']:
            self.viz = VisualizeSegAndClass(self,grid=(2,2))
        elif self.config['visualize_segs']:
            self.vizseg = VisualizeSegAndClass(self,grid=(1,2),figsize=(18,9))
        if self.config['debug']:
            logger.setLevel(logging.DEBUG)
        self.md = Metadata()


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

    def preprocess(self, img, normalize=True):
        if isinstance(img, str):
            img = skimage.io.imread(img)
        if np.prod(np.array(self.config['binning'])) != 1:
            img = downscale_local_mean(img, self.config['binning'])

        img = (img.astype(np.single))[:,1:]

        if not normalize:
            return np.stack([img, img, img], axis=-1) #np_pixels_downscaled
        return CellClassifier.normalize_image(img)

    @staticmethod
    def normalize_image(img, as_gray=False,mask=None):
        if mask is not None:
            img[img == mask] = np.nan

        # Normalize images
        percentile = 99.9
        high = np.nanpercentile(img, percentile)
        low = np.nanpercentile(img, 100 - percentile)

        img = np.minimum(high, img)
        img = np.maximum(low, img)

        img = np.minimum(1,(img - low) / (high - low))   # (0->1)

        # Make a RGB-channel color image
        if not as_gray: img = np.stack([img, img, img], axis=-1)
        return img

    def segment_nucleus(self,img_name):
        '''
        Segment nuclei using the 3rd place segmenting algorithm from the Kaggle Data Science Bowl 2018
        :param img_name: File name of the microscope image to process
        :return: Dictionary: {'model_data':results, 'molded_image':molded_image, 'meta':meta}
                 where results is the dictionary returned by the DSB segmenting algorithm,
                 'molded_image' is the submitted image after processing, and meta is metadata.
        '''

        t0 = TimeIt()
        img = skimage.io.imread(img_name,as_gray=True)
        full_path = str(Path(img_name).absolute())
        img = self.preprocess(img)
        if self.viz: self.viz.visualize_image(img, 'Dendra2 image (segment)')
        if self.vizseg: self.vizseg.visualize_image(img, 'Dendra2 image (segment)')
        image_id = random.randint(0, 1<<31)



        # Lazily initialize segmentation model
        inference_config = self.inference_config

        # Mold image
        molded_image, window, scale, padding, crop = utils.resize_image(img,
                                                                        min_dim=inference_config.IMAGE_MIN_DIM,
                                                                        min_scale=inference_config.IMAGE_MIN_SCALE,
                                                                        max_dim=inference_config.IMAGE_MAX_DIM,
                                                                        mode=inference_config.IMAGE_RESIZE_MODE)
        active_class_ids = [1, 1]
        meta = {
            'image_id' : random.getrandbits(64),
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
        model = self.model()

        t0 = TimeIt()
        image_metas = modellib.compose_image_meta(image_id, img.shape, molded_image.shape, window,
                                                  scale, active_class_ids).astype(np.int32)
        # Model trained on 0->255 and we have 0->1 in our data, so need a multiply here
        results = model.detect_molded(np.expand_dims(molded_image*255., 0), np.expand_dims(image_metas, 0), verbose=0)
        meta['segment_time'] = t0.get_time()
        meta['ncells'] = 0
        if len(results) > 0:
            meta['ncells'] = len(results[0]['class_ids'])
        logger.debug(f"{meta['ncells']} cells detected")
        logger.debug(f"metadata from segmentation {meta}")
        if self.viz: self.viz.visualize_cell_boundaries(results, molded_image, title='Segmented cell boundaries')
        if self.vizseg: self.vizseg.visualize_cell_boundaries(results, molded_image, title='Segmented cell boundaries')
        return {'model_data':results, 'molded_image':molded_image, 'meta':meta}


    @pickler
    def run(self, simg, cimg):
        '''
        Segment and classify a multichannel image
        :param simg: Image to use for segmentation
        :param cimg: Image to use for classification
        :param tf_gpu_fraction: Fraction of GPU allocated for TensorFlow. If none, default to
        default TF memory allocation
        :return:
        '''
        segmented = self.segment_nucleus(simg)
        rval = self.classify_image(cimg,segmented)
        self.md.add_row(segmented['meta'], rval['meta'],expts=self.config['expts'])
        return rval

    def classify_image(self, img_name, segmented):
        '''
        Classify data from previously segmented nuclei generated from cell images
        :param img_name: file name of the image to classify
        :param segmented: Return value from the segmented_nucleus function.
        :return: Dictionary: {'model_data':output_mask, 'meta':celldata} where the 'model_data' contains the
                            output mask generated by the classifier and 'meta' contains metadata from the
                            classification algorithm
        '''
        t0 = TimeIt()
        img = skimage.io.imread(img_name)
        full_path = str(Path(img_name).absolute())
        img = self.preprocess(img,normalize=False)
        if self.viz: self.viz.visualize_image(self.normalize_image(img[...,0]), 'LMNA image (classify)')

        # Mold image
        from mrcnn import utils
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
            bbox_slice = (slice(*cell_bb[i,[0,2]]), slice(*cell_bb[i,[1,3]]))
            mask = results[0]['masks'][i][bbox_slice]
            cell_size[i] = np.sum(mask,axis=(0,1))
            crop = molded_image[bbox_slice + (0,)]
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
            crop = np.pad(crop,((padsmin[0],padsmax[0]),(padsmin[1],padsmax[1])),'constant')

            # Crop if image is too large
            crop_max = crop_size - crop_min - cshapes
            crop_min = np.maximum(0,-crop_min)
            crop_max = np.maximum(0,-crop_max)
            crop = skimage.util.crop(crop,((crop_min[0],crop_max[0]),(crop_min[1],crop_max[1])))


            # Keep track of predict index to original index
            pred_to_index[len(list_of_crops)] = i
            # Normalize cell
            crop = self.normalize_image(crop)
            list_of_crops.append(np.moveaxis(crop, -1, 0))

        preprocess_time = t0.get_time()

        t0 = TimeIt()
        name = Path('_'.join(img_name.name.split('_')[-3:])).with_suffix('')
        if self.config['visualize_classifications']:
            vc = VisualizeClassifications(self.config, grid=(4, 4))
        predictions = self._pytorch_model(img_name,list_of_crops)
        if self.config['visualize_classifications']:
            vc.set_title(name)
            vc.visualize(predictions, list_of_crops, n_cells)
        classify_time = t0.get_time()

        # Construct output mask and save information about objects
        output_mask = np.zeros((nshape[0],nshape[1]), dtype=np.int8)
        image_number = img_meta['image_id']
        celldata = {}
        for j,i in pred_to_index.items():
            bbox_slice = (slice(*cell_bb[i,[0,2]]), slice(*cell_bb[i,[1,3]]))
            cclass = np.argmax(predictions[j]) + 1
            seg_mask = results[0]['masks'][i][bbox_slice] * cclass
            output_mask[bbox_slice] = np.bitwise_or(output_mask[bbox_slice],seg_mask)
            celldata[i] = dict(
                zip(
                    ('image_id','source_image_path','dims','centroid','cell_size',
                     'boundary_cell', 'activated','preprocess_time','classify_time'),
                    (image_number, full_path, cell_dim[i],
                                       cell_centroid[i], cell_size[i], boundary_cell[i],
                                       cclass,classify_time,preprocess_time)
                ))
        output_mask = output_mask[padding[0][0]:nshape[0] - padding[0][1], padding[1][0]:nshape[1] - padding[1][1]]
        if not output_mask.flags.c_contiguous:
            output_mask = output_mask.copy(order='C')
        if self.viz: self.viz.visualize_image(output_mask, 'Predictions')
        logger.debug(f"metadata from classification {celldata}")
        return {'model_data':output_mask, 'meta':celldata}

    def _pytorch_model(self, img_name, list_of_crops):
        # Perform classifier predictions on gpu
        # Initialize pytorch model
        n_cells = len(list_of_crops)
        logger.debug(f"Loading PyTorch classification model from {config['classification_model_path']}")
        from fastai.basic_data import DatasetType
        from fastai.basic_train import load_learner
        from fastai.torch_core import tensor
        from fastai.vision.data import normalize
        learn = load_learner(config['classification_model_path'], config['classification_model_file'])
        logger.debug(f"PyTorch model loaded")
        imagenet_mean = tensor([0.485, 0.456, 0.406])
        imagenet_std = tensor([0.229, 0.224, 0.225])
        list_of_crops = [torch.from_numpy(crop) for crop in list_of_crops]
        if n_cells > 0:
            predictions = []
            for k in range(0, n_cells, config['batch_size']):
                # [prob1,prob2,...,probn] per class
                results = learn.pred_batch(ds_type=DatasetType.Test,
                                 batch=(normalize(
                                     torch.stack(list_of_crops[k:k + config['batch_size']]).float(),
                                     imagenet_mean,
                                     imagenet_std).cuda(),
                                        tensor(range(n_cells))))
                #predict = np.argmax(results, axis=1).numpy() + 1
                predictions.append(results)
            predictions = np.concatenate(predictions)
        else:
            predictions = np.array([])
        return predictions

class Metadata(object):
    def __init__(self):
        self.md = pd.DataFrame()

    def add_row(self, seg_meta, class_meta, expts=None):
        segment_img = seg_meta['source_image_path']
        classify_img = class_meta[0]['source_image_path']
        d = {'image_id': seg_meta['image_id'],
             'segment_dir': Path(segment_img).parent,
             'classify_dir': Path(classify_img).parent,
             'segment_file': Path(segment_img).name,
             'classify_file': Path(classify_img).name,
         }

        rows = []
        for id,cmeta in enumerate(class_meta.values()):
            cell_d = {k:v for k,v in cmeta.items() if k in ('activated', 'boundary_cell', 'cell_size', 'centroid')}
            cell_d['cell_id'] = id
            nd = d.copy()
            nd.update(cell_d)
            rows.append(nd)
        metadata = pd.DataFrame(data=rows)
        metadata = metadata.join(
            metadata['classify_file'].str.extract(r'.*?_(?P<well>[A-Z][0-9][0-9])_s(?P<site>[0-9]?[0-9]?[0-9]?[0-9])_w'))
        if expts:
            for k, v in expts.items():
                metadata[k] = metadata['well'].apply(v)
        metadata = metadata.set_index('image_id')
        self.md = self.md.append(metadata)


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--visualize', help="Visualize results", action="store_true")
    group.add_argument('--visualize_classifications', help="Visualize classification results", action="store_true")
    group.add_argument('--visualize_segs', help="Visualize segmentation results", action="store_true")
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
        #'classification_model_path':'./models/imageset_divided/2x2_binning', #use hyeon-jin's resnet34 model trained on 2x2 binned crops with multiple classes,
        #'classification_model_file':'export.pkl',
        'classification_model_path' : '../classifier-images/imageset_3class',
        'classification_model_file':'model_64.20200229_1659.pkl',
        'save_every':500,
        'cell_size_minimum':0,
        'batch_size':512,
        'expand_pixels':2,
        'boundary_size':3,
        'crop_size':64,
        'visualize':args.visualize,
        'visualize_segs' : args.visualize_segs,
        'visualize_classifications' : args.visualize_classifications,
        'binning': (1,1),
        'debug': args.debug,
        'tf_gpu_fraction': args.tf_gpu_fraction,
        'expts': {
            'Treatment': lambda
                x: 'Bortezomib' if 'B' in x else 'None' if 'A' in x else 'Unknown' if 'C' in x else 'Bortezomib' if any(
                y in x for y in ['D04', 'D05', 'D06']) else 'None',
            'Variant': lambda
                x: 'Library' if 'D' in x else 'N195K' if '01' in x else 'E145K' if '02' in x else 'WT' if '03' in x else 'E358K' if '04' in x else 'R386K' if '05' in x else 'R482L'
        }
    }
    logger.info("Configuration:")
    logger.info(pprint.pformat(config))

    def validate_segment(fname):
        '''
        Make sure this really is a Dendra image (520nm)
        :param fname:
        :return:
        '''
        from skimage.external.tifffile import TiffFile
        with open(fname,"rb") as f:
            tfile = TiffFile(f)
            img = tfile.pages[0]
            match = re.search(r'id="wavelength".*?value="(\d+)"/',str(img.tags['image_description']))
            if match is None or len(match.groups()) < 2:
                logger.warning(f"Can't find metadata for wavelength. Continuing")
            elif int(match.group(1)) != 520:
                raise Exception(f"Invalid segmentation file: wavelength is {int(match.group(1))}nm")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        mon = CellClassifier(config)
        if args.action == 'segment':
            # Segment the images using the DSB MASK-RCNN and save results (if requested)
            files = sorted(list(Path(args.input_dir).glob(args.seg_pattern)))
            validate_segment(files[0])

            if args.max_images > 0: files = files[0:args.max_images]
            outfiles = None
            if args.seg_dir:
                outfiles = [(Path(args.seg_dir) / p.parts[-1]).with_suffix('.pkl') for p in files]
            files = tqdm.tqdm(files)
            for i,f in enumerate(files):
                #files.set_description(f"Segmenting {f}")
                results = mon.segment_nucleus(f)
                # Use sparse matrix for masks
                masks = results['model_data'][0]['masks']
                results['model_data'][0]['masks'] = [bsr_matrix(m) for m in masks]
                if outfiles is not None:
                    with open(outfiles[i],"wb") as f:
                       pickle.dump(results,f)
        elif args.action == 'classify':
            # Classify previously segmented images
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
            # Segment and classify images using DSB Mask-RCNN for segmentation + PyTorch Unet for classification
            cfiles = sorted(list(Path(args.input_dir).glob(args.classify_pattern)))
            sfiles = sorted(list(Path(args.input_dir).glob(args.seg_pattern)))
            if args.max_images > 0: cfiles = cfiles[0:args.max_images]
            if args.max_images > 0: sfiles = sfiles[0:args.max_images]
            assert len(cfiles) == len(sfiles)
            sfiles = tqdm.tqdm(sfiles)
            for i,sfile in enumerate(sfiles):
                results = mon.run(sfile, cfiles[i])
                # with (open("/tmp/bad.pkl","wb")) as f:
                #     pickle.dump(results,f)
            print(mon.md.md)
        else:
            # Should never reach here
            raise Exception(f"Invalid analysis type {args.action}")



