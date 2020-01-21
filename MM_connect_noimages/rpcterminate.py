import rpyc
import time
import os

def get_outputmask(image_name, output_mask):
    #....
    print("call back in client", image_name)

conn = rpyc.connect("localhost", 18871)
bgsrv = rpyc.BgServingThread(conn)  # create a bg thread to process incoming events
mon = conn.root.ImageAnalysis(get_outputmask) # create a filemon

mon.finalize()
bgsrv.stop()
conn.close()

