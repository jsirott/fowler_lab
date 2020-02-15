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



if __name__ == '__main__':
    # w1 is LMNA for 09-13-19_LMNA_variants_tile2_bortezomib_20X , usually w2. Sigh.
    create_tophat('../09-13-19_LMNA_variants_tile2_bortezomib_20X','../09-13-19_LMNA_variants_tile2_bortezomib_20X', segment_pattern='**/*w1.TIF', visualize=False, normalize=True)

    expts = {
        'Treatment': lambda x: 'Bortezomib' if 'B' in x else 'None' if 'A' in x else 'Unknown' if 'C' in x else 'Bortezomib' if any(
            y in x for y in ['D04', 'D05', 'D06']) else 'None',
        'Variant': lambda x: 'Library' if 'D' in x else 'N195K' if '01' in x else 'E145K' if '02' in x else 'WT' if '03' in x else 'E358K' if '04' in x else 'R386K' if '05' in x else 'R482L'
    }

    df = get_metadata('../09-13-19_LMNA_variants_tile2_bortezomib_20X',segment_pattern='**/*w2.TIF', classify_pattern='**/*w1.TIF', expts=expts)
