import numpy as np
import skimage
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
from fastai.imports import *
from fastai.vision import *
import datetime as dt
import sys
import skimage.io
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


np.random.seed(2)

def train(ckpt=None,cycles=100, max_lr = slice(None,1e-2,None),freeze=True,metric=accuracy,
          model = models.resnet50,do_lr_find=False,size=None,bs=64):
    logger.info(f"Training model {model} with size of {size}")
    ckptfile = f"model_{size}.{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
    data = get_data(size=size,bs=bs)
    logger.info(f"Data is {data}")
    learn = cnn_learner(data, model, metrics=metric)
    if do_lr_find:
        learn.lr_find(stop_div=False, num_it=50)
        learn.recorder.plot()
        plt.show()
        return
    if freeze:
        learn.freeze()
    else:
        learn.unfreeze()
    if ckpt:
        learn.load(ckpt)
    learn.fit_one_cycle(cycles, max_lr=max_lr,callbacks=[EarlyStoppingCallback(learn,patience=30)])
    learn.save(ckptfile)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_top_losses(25, figsize=(15, 11))
    interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
    plt.show()
    return ckptfile

def export(ckpt,model,size,bs):
    data = get_data(size=size,bs=bs)
    learner = cnn_learner(data,model)
    learner.load(ckpt)
    learner.export(ckpt)

class MyTransform(TfmPixel):
    '''
    Hack to get added transforms to work in fast.ai
    Example:
        def _identity(x):
            skimage.io.imshow(np.moveaxis(x.numpy(),0,-1))
            plt.show()
            return x
        identity = MyTransform(_identity)
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tfm = self

    def resolve(self):
        pass

def get_data(size,bs):
    xforms = get_transforms(do_flip=True, max_rotate=180)
    data = ImageDataBunch.from_folder(path='../classifier-images/imageset_divided_tophat/',
                                      train='train', valid='validation',
                                      ds_tfms=xforms,
                                      size=data_size,bs=bs)
    data.show_batch(rows=3, figsize=(7, 6))
    data.normalize(imagenet_stats)
    return data

if __name__ == '__main__':
    data_size = 256  # 1,1 binning
    bs = 64
    lr = 7e-3
    model = models.resnet34
    #train(cycles=1000,model=models.resnet50,max_lr = slice(None,7e-3,None),do_lr_find=False,size=data_size,bs=bs)
    #train(ckpt='model_test_64.20200120_1303',cycles=1000, freeze=False, model=models.resnet50, max_lr=slice(None, 7e-3, None), do_lr_find=False, size=data_size,bs=bs)
    #export(ckpt='model_test_64.20200120_1303',model=models.resnet50, size=data_size,bs=bs)
    rval = train(cycles=1000,model=model,max_lr = slice(None,7e-3,None),do_lr_find=False,size=data_size,bs=bs)
    print(rval)
    train(ckpt=rval.replace('.pth',''),cycles=1000, freeze=False, model=model, max_lr=slice(None, 7e-3, None), do_lr_find=False, size=data_size,bs=bs)
    export(ckpt=rval.replace('.pth',''),model=model, size=data_size,bs=bs)

