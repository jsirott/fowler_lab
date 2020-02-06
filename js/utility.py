import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat,disk,square
import numpy as np
from scipy import ndimage as ndi




def create_tophat(indir,outdir,pattern=None):
    pattern = '**/*.png' if pattern is None else pattern
    indir = Path(indir)
    outdir = Path(outdir)
    if not indir.is_dir():
        raise Exception(f"{indir} is not a directory")
    outdir.mkdir(parents=True,exist_ok=True)
    for i in indir.glob(pattern):
        nfile = outdir.joinpath(*i.parts[-3:])
        nfile.parent.mkdir(parents=True,exist_ok=True)
        print(i,nfile)
        img = skimage.io.imread(i,as_gray=True).astype(np.float32)
        th = white_tophat(img, square(10))
        #nimage = np.moveaxis(np.stack((img, th, np.zeros(img.shape))), 0, -1)
        nimage = np.moveaxis(np.stack((th, th, th)), 0, -1)
        #visualize_tophat(img,th,nimage)
        skimage.io.imsave(nfile,(nimage*255).astype(np.uint8))



def visualize_tophat(img,th,combined):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(th, cmap='gray')
    ax[2].imshow(combined)
    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    create_tophat('../classifier-images/imageset_divided','../classifier-images/imageset_divided_tophat',pattern='**/*.png')