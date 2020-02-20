import os
import sys

import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.morphology import white_tophat,disk,square
import numpy as np
from cell_classifier import CellClassifier
import pandas as pd
import logging
# Setup logging
from tqdm import tqdm
import re
import pprint
from mrcnn import utils
from skimage.filters import threshold_otsu
from copy import copy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('max_colwidth',500)
pd.set_option('max_columns', 100)
pd.set_option('display.width',1000)

def get_metadata(dir,segment_pattern,classify_pattern,expts=None,save=True):
    if not Path(dir).is_dir():
        raise Exception(f"{dir} is not a directory")
    segment_images = list(Path(dir).absolute().glob(segment_pattern))
    classify_images = list(Path(dir).absolute().glob(classify_pattern))
    assert len(segment_images) == len(classify_images)
    metadata = pd.DataFrame(
        data={'image_number': list(range(1, len(segment_images) + 1)),
              'segment_path': segment_images,
              'classify_path': classify_images,
              'segment_base': [p.name for p in segment_images],
              'classify_base': [p.name for p in classify_images],
              })
    metadata = metadata.join(metadata['segment_base'].str.extract(r'.*?_(?P<image_metadata_well>[A-Z][0-9][0-9])_s(?P<image_metdata_site>[0-9]?[0-9]?[0-9]?[0-9])_w'))
    if expts:
        for k,v in expts.items():
            metadata[k] = metadata['image_metadata_well'].apply(v)
    metadata = metadata.set_index('image_number')
    if save: metadata.to_csv(Path(dir).joinpath('metadata.csv'))
    return metadata

def create_tophat(indir, outdir, segment_pattern=None, visualize=False, normalize=False):
    segment_pattern = '**/*.png' if segment_pattern is None else segment_pattern
    indir = Path(indir).resolve()
    outdir = Path(outdir).resolve()
    if not indir.is_dir():
        raise Exception(f"{indir} is not a directory")
    outdir.mkdir(parents=True,exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12,12))
    files = tqdm(list(indir.glob(segment_pattern)))

    for i in files:
        outfile = re.sub(r'w\d.TIF','tophat.TIF',i.parts[-1])
        nfile = outdir.joinpath(outfile)
        nfile.parent.mkdir(parents=True,exist_ok=True)
        img = skimage.io.imread(i,as_gray=True).astype(np.float32)
        if normalize: img = CellClassifier.normalize_image(img,as_gray=True)
        th = white_tophat(img, square(10))
        if visualize: visualize_tophat(img,th,fig,ax,i)
        skimage.io.imsave(nfile,(th*255).astype(np.uint8))

def visualize_tophat(img,th,fig,ax,title=None):
    if title: fig.suptitle(title)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(th, cmap='gray')
    ax[1].set_title('Tophat')
    fig.set_tight_layout(True)
    plt.draw()
    plt.waitforbuttonpress(60*30)

def mask_images(config, indir, seg_pattern, class_pattern, class_orig_pattern):
    # Get the segmented regions
    # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    seg_files = sorted(list(Path(indir).glob(seg_pattern)))
    class_files = sorted(list(Path(indir).glob(class_pattern)))
    class_orig_files = sorted(list(Path(indir).glob(class_orig_pattern)))
    assert len(seg_files) == len(class_files) == len(class_orig_files)
    fig, axes = plt.subplots(2, 3, figsize=(16,16))
    axes = axes.ravel()
    classifier = CellClassifier(config)
    start = 10
    for i,f in enumerate(seg_files[start:]):
        class_file = class_files[i]
        class_orig_file = class_orig_files[i]
        print(f,class_file, class_orig_file)

        img1 = classifier.preprocess(str(class_file), normalize=False)
        img1 = img1/255.
        img1, _, _, _, _ = utils.resize_image(img1,
                                              min_dim=classifier.inference_config.IMAGE_MIN_DIM,
                                              min_scale=classifier.inference_config.IMAGE_MIN_SCALE,
                                              max_dim=classifier.inference_config.IMAGE_MAX_DIM,
                                              mode=classifier.inference_config.IMAGE_RESIZE_MODE)

        img_orig = classifier.preprocess(str(class_orig_file), normalize=True)
        img_orig, _, _, _, _ = utils.resize_image(img_orig,
                                              min_dim=classifier.inference_config.IMAGE_MIN_DIM,
                                              min_scale=classifier.inference_config.IMAGE_MIN_SCALE,
                                              max_dim=classifier.inference_config.IMAGE_MAX_DIM,
                                              mode=classifier.inference_config.IMAGE_RESIZE_MODE)

        results = classifier.segment_nucleus(f)
        model_data = results['model_data'][0]
        img2 = results['molded_image']
        for j, roi in enumerate(model_data['rois']):
            mask = model_data['masks'][j]

            cell_bb = roi.astype(np.int32)
            bbox_slice = (slice(*cell_bb[[0, 2]]), slice(*cell_bb[[1, 3]]))

            sliced = img1*np.stack((mask,mask,mask),axis=-1)
            sliced = sliced[bbox_slice]
            visualize_tophat_in_context(img_orig, img1, sliced, img2, axes, cell_bb)


def visualize_tophat_in_context(img_orig, img_normalized, tophat_img, img_to_segment, axes, cell_bb):
    if np.amax(tophat_img) > 0.4:
        rect = patches.Rectangle(cell_bb[[1, 0]], *(cell_bb[[3, 2]] - cell_bb[[1, 0]]), edgecolor='r', facecolor='none')
        axes[0].set_title('Tophat extraction')
        axes[0].imshow(tophat_img)
        axes[1].set_title('LMNA-Tophat segment box')
        axes[1].imshow(img_normalized)
        axes[1].add_patch(copy(rect))
        axes[2].set_title('Dendra segment box')
        axes[2].imshow(img_to_segment)
        axes[2].add_patch(copy(rect))
        axes[3].set_title('LMNA segment box')
        axes[3].imshow(img_orig)
        axes[3].add_patch(copy(rect))
        axes[4].set_title('Tophat extraction histogram')
        axes[4].set_xlim(0, 1)
        axes[4].set_yscale('log')
        axes[4].hist(tophat_img.ravel())
        plt.tight_layout()
        plt.draw()
        plt.waitforbuttonpress(60 * 30)
        [a.clear() for a in axes]

if __name__ == '__main__':
    root_dir = "../DSB_2018-master/"
    model_file = "deepretina_final.h5"
    config = {
        'root_dir' : root_dir,
        'model_file': model_file,
        'model_path':os.path.join(root_dir, model_file),
        'classification_model_path':'./models/imageset_divided/2x2_binning', #use hyeon-jin's resnet34 model trained on 2x2 binned crops with multiple classes,
        'classification_model_file':'export.pkl',
        'save_every':500,
        'cell_size_minimum':0,
        'batch_size':512,
        'expand_pixels':2,
        'boundary_size':3,
        'crop_size':64,
        'visualize':False,
        'visualize_segs' : False,
        'visualize_classifications' :False,
        'binning': (1,1),
        'debug':True,
        'tf_gpu_fraction': None
    }
    mask_images(config, '../09-13-19_LMNA_variants_tile2_bortezomib_20X', '**/*B*w2*.TIF', '**/*B*tophat*.TIF', '**/*B*w1*.TIF')

    # if False:
    #     # w1 is LMNA for 09-13-19_LMNA_variants_tile2_bortezomib_20X , usually w2. Sigh.
    #     create_tophat('../09-13-19_LMNA_variants_tile2_bortezomib_20X','../09-13-19_LMNA_variants_tile2_bortezomib_20X', segment_pattern='**/*w1.TIF', visualize=False, normalize=True)
    #
    #     expts = {
    #         'Treatment': lambda x: 'Bortezomib' if 'B' in x else 'None' if 'A' in x else 'Unknown' if 'C' in x else 'Bortezomib' if any(
    #             y in x for y in ['D04', 'D05', 'D06']) else 'None',
    #         'Variant': lambda x: 'Library' if 'D' in x else 'N195K' if '01' in x else 'E145K' if '02' in x else 'WT' if '03' in x else 'E358K' if '04' in x else 'R386K' if '05' in x else 'R482L'
    #     }
    #
    #     df = get_metadata('../09-13-19_LMNA_variants_tile2_bortezomib_20X',segment_pattern='**/*w2.TIF', classify_pattern='**/*w1.TIF', expts=expts)
