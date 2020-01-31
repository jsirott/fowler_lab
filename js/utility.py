import glob
import skimage.io
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import white_tophat, square
import numpy as np



def create_tophat(indir,outdir):
    indir = Path(indir)
    outdir = Path(outdir)
    if not indir.is_dir():
        raise Exception(f"{indir} is not a directory")
    outdir.mkdir(parents=True,exist_ok=True)
    for i in indir.glob('**/*.png'):
        nfile = outdir.joinpath(*i.parts[-3:])
        nfile.parent.mkdir(parents=True,exist_ok=True)
        print(i,nfile)
        img = skimage.io.imread(i,as_gray=True)
        th = white_tophat(img, square(10))
        nimage = np.moveaxis(np.stack((img, th, np.zeros(img.shape))), 0, -1)
        #visualize_tophat(img,nimage)
        skimage.io.imsave(nfile,(nimage*255).astype(np.uint8))



def visualize_tophat(img,th):
    fix, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap='gray')
    #ax[1].imshow(th, cmap='gray')
    ax[1].imshow(th)
    plt.show()


if __name__ == '__main__':
    create_tophat('../classifier-images/imageset_divided','../classifier-images/imageset_divided_tophat')