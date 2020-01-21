
import os
import sys
import numpy as np
import skimage.io 
sys.path.append(os.path.abspath("/home/srirampendyala/Projects/DSB_2018-master/mrcnn/"))
from utils import Dataset

class BowlDataset(Dataset):

    def load_bowl(self, folderpaths):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("bowl", 1, "nucleus")
        self.add_class("bowl", 2, "micronucleus_intact")
        self.add_class("bowl", 3, "micronucleus_burst")
        self.add_class("bowl", 4, "outoffocus")
        self.add_class("bowl", 5, "micronucleus_connected")

        #Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc. 

        # Add images
        for i in range(len(folderpaths)):
            self.add_image("bowl", image_id=i, path=folderpaths[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image_path = os.path.join(image_path, 'images', '{}.png'.format(os.path.basename(image_path)))
        image = skimage.io.imread(image_path)
        image = image[:,:,:3]
        return image
        
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bowl":
            return info["id"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id, shape):
        """Generate instance masks for shapes of the given image ID.
        """
        
        info = self.image_info[image_id]
        image_path = info['path']
        mask_path = os.path.join(image_path, 'masks')
        class_id = {}
        masks = {}
        for i in range(1,len(self.class_names)):
            class_name = self.class_names[i]
            class_path = os.path.join(mask_path, class_name)
            if os.path.exists(class_path):
                if len(os.listdir(class_path)) > 0:
                    class_masks = skimage.io.imread_collection(os.path.join(class_path, '*.png')).concatenate()
                    class_masks = np.rollaxis(class_masks,0,3)
                    class_masks = np.clip(class_masks,0,1)
                    masks[class_name] = class_masks
                    class_id[class_name] = [self.class_ids[i]]*class_masks.shape[2]
        if len(masks.keys()) > 0:
            class_id = np.concatenate([class_id[key] for key in masks.keys()], axis=-1)
            masks = np.concatenate([masks[key] for key in masks.keys()], axis=-1)
        else:
            masks = np.zeros((shape[0],shape[1],1))
            class_id = np.array([1])
        return masks, class_id.astype(np.int32)

