import os
import sys

import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(2, 2, figsize=(16,16))
    ax = ax.ravel()
    cc = CellClassifier(config)
    for i,f in enumerate(seg_files):
        class_file = class_files[i]
        class_orig_file = class_orig_files[i]
        print(f,class_file, class_orig_file)

        img1 = cc.preprocess(str(class_file), normalize=False)
        img1 = img1/255.
        img1, _, _, _, _ = utils.resize_image(img1,
                                              min_dim=cc.inference_config.IMAGE_MIN_DIM,
                                              min_scale=cc.inference_config.IMAGE_MIN_SCALE,
                                              max_dim=cc.inference_config.IMAGE_MAX_DIM,
                                              mode=cc.inference_config.IMAGE_RESIZE_MODE)

        img_orig = cc.preprocess(str(class_orig_file), normalize=True)
        img_orig, _, _, _, _ = utils.resize_image(img_orig,
                                              min_dim=cc.inference_config.IMAGE_MIN_DIM,
                                              min_scale=cc.inference_config.IMAGE_MIN_SCALE,
                                              max_dim=cc.inference_config.IMAGE_MAX_DIM,
                                              mode=cc.inference_config.IMAGE_RESIZE_MODE)

        results = cc.segment_nucleus(f)
        model_data = results['model_data'][0]
        img2 = results['molded_image']
        for j, roi in enumerate(model_data['rois']):

            from skimage.draw import rectangle, rectangle_perimeter
            cell_bb = roi.astype(np.int32)
            bbox_slice = (slice(*cell_bb[[0, 2]]), slice(*cell_bb[[1, 3]]))

            rr, cc = rectangle_perimeter(cell_bb[[0, 1]], cell_bb[[2, 3]] - 2)
            masked = np.copy(img1)
            masked[rr,cc,1:] = 0
            masked[rr,cc,0] = 1

            masked2 = np.copy(img2)
            masked2[rr, cc, 1:] = 0
            masked2[rr, cc, 0] = 1

            masked3 = np.copy(img_orig)
            masked3[rr, cc, 1:] = 0
            masked3[rr, cc, 0] = 1

            sliced = img1[bbox_slice]
            if np.amax(sliced) > 0.5:
                ax[0].set_title('Tophat extraction')
                ax[0].imshow(sliced)
                ax[1].set_title('LMNA-Tophat segment box')
                ax[1].imshow(masked)
                ax[2].set_title('Dendra segment box')
                ax[2].imshow(masked2)
                ax[3].set_title('LMNA segment box')
                ax[3].imshow(masked3)
                plt.tight_layout()
                plt.draw()
                plt.waitforbuttonpress(60 * 30)

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
    mask_images(config, '../09-13-19_LMNA_variants_tile2_bortezomib_20X', '**/*w2*.TIF', '**/*tophat*.TIF', '**/*w1*.TIF')

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
