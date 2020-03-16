#!/usr/bin/env python

import argparse

import rpyc
import os
from rpyc.utils.server import ThreadPoolServer
from rpyc.utils.helpers import classpartial

import pprint
import logging
from cell_classifier import CellClassifier
import pickle
import numpy as np
import time
import client
import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RpycServer(rpyc.Service):
    def __init__(self,config):
        logger.info(f"Starting up new RpycServer class")
        self.analyzer = CellClassifier(config=config)

    def exposed_run(self, *args, **kwargs):
        return self.analyzer.run(*args, **kwargs)

    def exposed_write_metadata(self, *args, **kwargs):
        return self.analyzer.write_metadata(*args, **kwargs)


class ImageAnalysisServer:
    def __init__(self,config,hostname='localhost',port=10000):
        service = classpartial(RpycServer, config)
        # GPU can only handle one request at a time, so nbThreads has to equal 1
        self.server = ThreadPoolServer(service, hostname=hostname, port=port, nbThreads=1)

    def start(self):
        self.server.start()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment and/or classify an image')
    parser.add_argument('--hostname', help="host to bind to", default='localhost')
    parser.add_argument('--port', help='TCP port to bind to', default=10000)
    parser.add_argument('--test', help='Run one test example and exit', action='store_true')
    parser.add_argument('--debug', help='Log debugging info', action='store_true')
    args = parser.parse_args()
    port = args.port
    hostname = args.hostname
    # Set configuration
    root_dir = "../DSB_2018-master/"
    model_file = "deepretina_final.h5"
    config = {
        'root_dir': root_dir,
        'model_file': model_file,
        'model_path': os.path.join(root_dir, model_file),
        'classification_model_path':'./models/imageset_divided/2x2_binning', #use hyeon-jin's resnet34 model trained on 2x2 binned crops with multiple classes,
        'classification_model_file':'export.pkl',
        'save_every': 500,
        'cell_size_minimum': 0,
        'batch_size': 512,
        'expand_pixels': 2,
        'boundary_size': 3,
        'crop_size': 64,
        'visualize_segs': False,
        'visualize_classifications': False,
        'binning': (2, 2),
        'debug': args.debug,
        'tf_gpu_fraction': 0.75
    }
    logger.info("Configuration:")
    logger.info(pprint.pformat(config))

    server = ImageAnalysisServer(config)

    if args.test:
        try:
            # Get saved results
            import multiprocessing as mp
            logger.info("Test mode. Forking server")
            def start_server():
                server.start()
            p = mp.Process(target = start_server)
            p.start()
            time.sleep(1)

            client = client.CellClassifierClient()
            client.connect(hostname, port)
            logger.info("Running non remote analysis")

            analysis = client.run('test/test_segment.tif', 'test/test_classify.tif')
            client.write_metadata('test')
            # with open('test/test.pkl',"wb") as f:
            #     pickle.dump(analysis,f)
            p.terminate()
            with open("test/test.pkl","rb") as f:
                real = pickle.load(f)
            assert np.array_equal(real['model_data'],analysis['model_data'])
            print('*********** Test passed *************')
        except Exception as e:
            print('*********** Test failed *************')
            raise e
    else:
        server.start()