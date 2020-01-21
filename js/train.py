import numpy as np
from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
from fastai.imports import *
from fastai.vision import *
import datetime as dt
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


np.random.seed(2)

def train(ckpt=None,cycles=100, max_lr = slice(None,1e-2,None),freeze=True,metric=accuracy,
          model = models.resnet50,do_lr_find=False,size=None,bs=64):
    logger.info(f"Training model {model} with size of {size}")
    ckptfile = f"model_test_{size}.{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
    data = get_data(size=size,bs=bs)
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
#        learn.export(f"{ckpt}.pkl")
#        sys.exit(1)
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


def get_data(size,bs):
    data = ImageDataBunch.from_folder(path='../classifier-images/imageset_divided/',
                                      train='train', valid='validation',
                                      ds_tfms=get_transforms(do_flip=True, max_rotate=180), size=data_size,bs=bs)
    data.show_batch(rows=3, figsize=(7, 6))
    data.normalize(imagenet_stats)
    return data

if __name__ == '__main__':
    data_size = 64  # 2,2 binning
    bs = 256
    #train(cycles=1000,model=models.resnet50,max_lr = slice(None,7e-3,None),do_lr_find=False,size=data_size,bs=bs)
    #train(ckpt='model_test_64.20200120_1303',cycles=1000, freeze=False, model=models.resnet50, max_lr=slice(None, 7e-3, None), do_lr_find=False, size=data_size,bs=bs)
    export(ckpt='model_test_64.20200120_1303',model=models.resnet50, size=data_size,bs=bs)

