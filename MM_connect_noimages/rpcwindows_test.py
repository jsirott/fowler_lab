import rpyc
import os
import sys
import numpy as np
import skimage.io
import base64

def get_outputmask(image_name, output_mask):
    #....
    print("call back in client", image_name)

if __name__ == "__main__":

    conn = rpyc.connect(“linuxhost”, 18871)
    bgsrv = rpyc.BgServingThread(conn)  # create a bg thread to process incoming events
    mon = conn.root.ImageAnalysis(get_outputmask) # create a filemon

    for imageno in range(1, 10):
       img = skimage.io.imread("")
       mon.run_pipeline(“image” + str(imageno), img):

    mon.finalize()
    bgsrv.stop()
    conn.close()
