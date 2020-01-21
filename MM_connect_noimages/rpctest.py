import rpyc
import os
import sys
import time
import numpy as np
import skimage.io
import base64

class LinuxService(rpyc.Service):
    class exposed_ImageAnalysis(object):
        def __init__(self, callback):

            self.callback = rpyc.async_(callback)   # make the callback async
            return

        def exposed_run_pipeline_on_image(self, image_name, img):
            
            img_reconstructed = np.frombuffer(base64.b64decode(img), np.uint16).reshape((image_height,image_width))
            time.sleep(45)
            #skimage.io.imsave(os.path.join(output_dir,image_name+'.png'), img_reconstructed)
            return base64.b64encode(img_reconstructed)

if __name__ == "__main__":

    #Set global variables
    output_dir = './test/'
    image_height = 1536
    image_width = 1740

    #Start rpyc server
    from rpyc.utils.server import ThreadedServer
    ThreadedServer(LinuxService, port = 18871).start()
