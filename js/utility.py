import os
import sys

import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import skew
from skimage import img_as_ubyte
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

def visualize_tophat(img,th,fig,title=None):
    fig, ax = plt.subplots(2, 3, figsize=(16,16))
    if title: fig.suptitle(title)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(th, cmap='gray')
    ax[1].set_title('Tophat')
    fig.set_tight_layout(True)
    plt.draw()
    plt.waitforbuttonpress(60*30)

def visualize_training_data(indir, pattern):
    data_files = sorted(list(Path(indir).glob(pattern)))
    for f in data_files:
        print(f)
        img = skimage.img_as_float32(skimage.io.imread(f))[...,0]
        # Eliminate masking artifacts
        img[img < 0.1] = np.nan
        img = CellClassifier.normalize_image(img)
        visualize_segmented(img)



def mask_images(config, indir, seg_pattern, class_pattern, outdir=None, visualize=True):
    '''
    Mask segmented nuclei and write the data to a directory
    :param config:
    :param indir:
    :param seg_pattern:
    :param class_pattern:
    :param class_tophat_pattern:
    :return:
    '''
    if outdir is None:
        outdir = Path(indir).joinpath('masked')
        outdir.mkdir(parents=True, exist_ok=True)
    seg_files = sorted(list(Path(indir).glob(seg_pattern)))
    class_files = sorted(list(Path(indir).glob(class_pattern)))
    assert len(seg_files) == len(class_files)
    classifier = CellClassifier(config)
    for i,f in enumerate(seg_files):
        class_file = class_files[i]
        print(f,class_file)

        class_file_processed = classifier.preprocess(str(class_file), normalize=False)
        class_file_processed, _, _, _, _ = utils.resize_image(class_file_processed,
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

            sliced = class_file_processed*np.stack((mask,mask,mask),axis=-1)
            sliced = classifier.normalize_image(sliced[bbox_slice],as_gray=True)
            if visualize: visualize_segmented(sliced, class_file_processed, img2, cell_bb)
            name = Path('_'.join(class_file.name.split('_')[-3:])).with_suffix('')
            suboutdir = outdir.joinpath(name)
            suboutdir.mkdir(exist_ok=True)
            outfile = suboutdir.joinpath(f"c{j:04}").with_suffix('.png')

            skimage.io.imsave(outfile, img_as_ubyte(sliced))
            logger.info(f"Wrote masked image {outfile}")


def visualize_segmented(clipped_img, class_img=None, img_to_segment=None, cell_bb=None, logged=False):
    f = visualize_segmented
    if not hasattr(f,'_fig'):
        f._fig, f._axes = plt.subplots(2, 2, figsize=(16, 16))
    fig,axes = f._fig, f._axes
    axes = axes.ravel()
    if cell_bb is not None:
        rect = patches.Rectangle(cell_bb[[1, 0]], *(cell_bb[[3, 2]] - cell_bb[[1, 0]]), edgecolor='r', facecolor='none')
    clipped_img_nz = clipped_img[clipped_img > 0]
    axes[0].set_title('LMNA extraction')

    axes[0].imshow(clipped_img)
    if img_to_segment is not None:
        axes[1].set_title('Dendra segment box')
        axes[1].imshow(img_to_segment)
        axes[1].add_patch(copy(rect))
    if class_img is not None:
        axes[2].set_title('LMNA segment box')
        axes[2].imshow(CellClassifier.normalize_image(class_img, as_gray=True))
        axes[2].add_patch(copy(rect))
    if not logged:
        display_clipped = clipped_img_nz
        axes[3].set_xlim(0, 1)
    else:
        display_clipped = np.log(1 + clipped_img_nz)
        axes[3].set_xlim(0, np.log(2))
    tstr = f'Histm={np.mean(display_clipped):.2} std={np.std(display_clipped):.2} ' \
           f'min={np.min(display_clipped):.2} max={np.max(display_clipped):.2} skew={skew(display_clipped):.2})'
    axes[3].set_title(tstr)
    axes[3].set_yscale('log')
    axes[3].hist(display_clipped,bins=25)
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
    #mask_images(config, '../09-13-19_LMNA_variants_tile2_bortezomib_20X', '**/*w2*.TIF', '**/*w1*.TIF', '**/*tophat*.TIF')
    #mask_images(config, '../09-13-19_LMNA_variants_tile2_bortezomib_20X', '**/*D06*w2*.TIF', '**/*D06*w1*.TIF')
    visualize_training_data('../classifier-images/imageset_divided/train', '**/edge/*.png')

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
