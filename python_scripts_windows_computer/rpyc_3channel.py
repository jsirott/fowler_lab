import sys
import time
import copy
import numpy as np
import rpyc
import base64
import clr, System
import ctypes
from System import Array, Byte
from System.Runtime.InteropServices import GCHandle, GCHandleType


def Startup(param):
    Docommand(param)

def Docommand(param):
    start0 = time.time()
    
    imageHandle = 0
    imageName = ""
    imageHeight = 0
    imageWidth = 0
    imageDepth = 0

    writeTimerMessages = True

    # Gather information about the original image
    _,imageHandle = mm.GetCurrentImage(imageHandle)

    _,_,imageName = mm.GetImageName(imageHandle, imageName)
    _,_,imageHeight = mm.GetHeight(imageHandle, imageHeight)
    _,_,imageWidth = mm.GetWidth(imageHandle, imageWidth)
    _,_,imageDepth = mm.GetDepth(imageHandle, imageDepth)

    # Get pixels from the original image
    start = time.time()
    pixels = mm.GetImagePixels(imageHandle)
    end = time.time()

    if writeTimerMessages:
        mm.PrintMsg("Get Image Timer: " + str((end-start)*1000))

    # Get height and width
    width = pixels.GetLength(0)
    height = pixels.GetLength(1)
    if writeTimerMessages:
        mm.PrintMsg("Image Info: " + imageName + " " + str(width) + "x" + str(height) + " with depth " + str(imageDepth))
    
    # Convert to numpy array
    start = time.time()
    sourceHandle = GCHandle.Alloc(pixels, GCHandleType.Pinned)
    sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
    if imageDepth == 48:
        np_pixels = np.empty((height,width), order='F', dtype='uint16')
    elif imageDepth == 24:
        np_pixels = np.empty((height,width), order='F', dtype='uint8')
    destPtr = np_pixels.__array_interface__['data'][0]
    ctypes.memmove(destPtr, sourcePtr, np_pixels.nbytes)
    np_pixels = np_pixels.copy(order='C')
    np_pixels_phased = np.stack([np_pixels[:,np.arange(0,width,3)], np_pixels[:,np.arange(1,width,3)], np_pixels[:,np.arange(2,width,3)]], axis=-1)
    np_pixels_phased = np_pixels_phased.copy(order='C')
    width = width // 3
    end = time.time()

    if writeTimerMessages:
         mm.PrintMsg("Convert to numpy array timer: " + str((end-start)*1000))

    # Send to GPU server
    start = time.time()
    conn = rpyc.connect("128.208.8.66", 18871)
    conn._config['sync_request_timeout'] = 10000 #set timeout to 10k seconds
    bgsrv = rpyc.BgServingThread(conn)
    mon = conn.root.ImageAnalysis(get_outputmask)
    b64_output_mask = mon.run_pipeline(imageName, base64.b64encode(np_pixels_phased))
    np_output_mask = 255*(np.frombuffer(base64.decodebytes(b64_output_mask), np.bool).reshape((height,width)).astype(np.uint8))
    bgsrv.stop()
    conn.close()
    end = time.time()

    if writeTimerMessages:
         mm.PrintMsg("Send and receive time: " + str((end-start)*1000))

    # Convert back from numpy array
    start = time.time()
    if not np_output_mask.flags.f_contiguous:
        np_output_mask = np_output_mask.copy(order='F')
    output_mask = Array.CreateInstance(System.Byte, width, height)
    destHandle = GCHandle.Alloc(output_mask, GCHandleType.Pinned)
    sourcePtr = np_output_mask.__array_interface__['data'][0]
    destPtr = destHandle.AddrOfPinnedObject().ToInt64()
    ctypes.memmove(destPtr, sourcePtr, np_output_mask.nbytes)
    end=time.time()

    if writeTimerMessages:
         mm.PrintMsg("Convert from numpy array timer: " + str((end-start)*1000))

    # Generate a new image
    newImageHandle = 0
    newImageName = imageName + "_Binary"
    _,_,_,_,_,newImageHandle = mm.CreateImage(width, height, 8, newImageName, newImageHandle)

    # Write image
    start = time.time()
    _ = mm.WriteImage(newImageHandle, 0, 0, width, height, 8, 0, 0, output_mask)
    end = time.time()
    end0 = time.time()

    if writeTimerMessages:
        mm.PrintMsg("Save Image Timer: " + str((end-start)*1000))
        mm.PrintMsg("Total Time: " + str((end0-start0)*1000))
        mm.PrintMsg("--------------------------------------------------------------------")

def Shutdown():
    pass

def get_outputmask(image_name, output_mask):
    #....
    print("call back in client", image_name)
