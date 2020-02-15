import sys

import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat,disk,square
import numpy as np
from cell_classifier import CellClassifier
import pandas as pd
pd.set_option('max_colwidth',500)
pd.set_option('max_columns', 100)
pd.set_option('display.width',1000)

def get_metadata(dir,segment_pattern,classify_pattern):
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
    print(metadata['segment_base'].head())
    metadata = metadata.join(metadata['segment_base'].str.extract(r'.*?_(?P<Image_Metadata_Well>[A-Z][0-9][0-9])_s(?P<Image_Metadata_Site>[0-9]?[0-9]?[0-9]?[0-9])_w'))
    metadata['Treatment'] = metadata['Image_Metadata_Well'].apply(
        lambda x: 'Bortezomib' if 'B' in x else 'None' if 'A' in x else 'Unknown' if 'C' in x else 'Bortezomib' if any(
            y in x for y in ['D04', 'D05', 'D06']) else 'None')
    metadata['Variant'] = metadata['Image_Metadata_Well'].\
        apply(lambda x: 'Library' if 'D' in x else 'N195K' if '01' in x else 'E145K' if '02' in x else 'WT' if '03' in x else 'E358K' if '04' in x else 'R386K' if '05' in x else 'R482L')
    metadata = metadata.set_index('image_number')
    print(metadata.head())


def create_tophat(indir,outdir,pattern=None,visualize=False,normalize=False):
    pattern = '**/*.png' if pattern is None else pattern
    indir = Path(indir).resolve()
    outdir = Path(outdir).resolve()
    if not indir.is_dir():
        raise Exception(f"{indir} is not a directory")
    outdir.mkdir(parents=True,exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(24,12))
    for i in indir.glob(pattern):
        nfile = outdir.joinpath(*i.parts[-2:])
        nfile.parent.mkdir(parents=True,exist_ok=True)
        print(i,nfile)
        img = skimage.io.imread(i,as_gray=True).astype(np.float32)
        if normalize: img = CellClassifier.normalize_image(img,as_gray=True)
        th = white_tophat(img, square(10))
        #nimage = np.moveaxis(np.stack((img, th, np.zeros(img.shape))), 0, -1)
        nimage = np.moveaxis(np.stack((th, th, th)), 0, -1)
        if visualize: visualize_tophat(img,th,nimage,fig,ax,i)
        skimage.io.imsave(nfile,(nimage*255).astype(np.uint8))



def visualize_tophat(img,th,combined,fig,ax,title=None):
    if title: fig.suptitle(title)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(th, cmap='gray')
    ax[1].set_title('Tophat')
    ax[2].imshow(combined)
    ax[2].set_title('Tophat RGB')
    fig.set_tight_layout(True)
    plt.draw()
    plt.waitforbuttonpress(60*30)



if __name__ == '__main__':
    #create_tophat('../classifier-images/imageset_divided','../classifier-images/imageset_divided_tophat',pattern='**/*.png')
    #create_tophat('../classifier-images/imageset_divided','/tmp/crud',pattern='**/puncta/*.png',visualize=True)
    # Lots of following are out of focus
    # w1 is LMNA for 09-13-19_LMNA_variants_tile2_bortezomib_20X , usually w2. Sigh.
    #create_tophat('../09-13-19_LMNA_variants_tile2_bortezomib_20X','/tmp/crud',pattern='**/*w1.TIF',visualize=True,normalize=True)
    get_metadata('../09-13-19_LMNA_variants_tile2_bortezomib_20X',segment_pattern='**/*w2.TIF', classify_pattern='**/*w1.TIF')
