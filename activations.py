import os
import logging
from abc import ABCMeta
from abc import abstractmethod
from multiprocessing import dummy as multiprocessing
import os.path
import numpy as np
import PIL.Image
from tcav_utils import get_acts_key
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.Resize((185, 185)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
])

transform_pretrained = transforms.Compose([
    transforms.Resize((315, 315)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])


class ActivationInterface(object):
    """An abstract interface for processing and loading neural network activations and accessing the model."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def process_and_load_activations(self, bottleneck_names, concepts):
        pass

    @abstractmethod
    def get_model(self):
        pass

class ActivationBase(ActivationInterface):

    """A base class for generating and processing activations with a neural network model."""

    def __init__(self, model, acts_dir, custom=True,max_examples=1000):
        self.model = model
        self.acts_dir = acts_dir
        self.max_examples = max_examples
        self.data_filename = None
        self.custom=custom

    def get_model(self):
        return self.model

    @abstractmethod
    def get_examples_for_concept(self, concept):
        pass

    def get_activations_for_concept(self, concept, bottleneck):
        examples = self.get_examples_for_concept(concept)
        return self.get_activations_for_examples(examples, bottleneck)

    def get_activations_for_examples(self, examples, bottleneck):
        # TODO: examples.shape
        acts = self.model.run_examples(examples, bottleneck)
        return self.model.reshape_activations(acts).squeeze()

    def process_and_load_activations(self, bottleneck_names, concepts):
        acts = {}
        if self.acts_dir and not os.path.exists(self.acts_dir):
            os.makedirs(self.acts_dir)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in bottleneck_names:
                acts_path = os.path.join(
                    self.acts_dir, get_acts_key(concept, self.model.model_name, bottleneck_name)) if self.acts_dir else None
                if acts_path and os.path.exists(acts_path):
                    with open(acts_path, 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(f, allow_pickle=False).squeeze()
                        logging.info('Loaded {} shape {}'.format(
                            acts_path, acts[concept][bottleneck_name].shape))
                else:
                    acts[concept][bottleneck_name] = self.get_activations_for_concept(
                        concept, bottleneck_name)
                    if acts_path:
                        logging.info('{} does not exist, Making one...'.format(
                            acts_path))
                        with open(acts_path, 'wb') as f:
                            np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
        return acts


class ImageActivationGenerator(ActivationBase):
    """A class for generating activations from image data for a neural network model."""

    def __init__(self, model, source_dir, acts_dir,custom, max_examples=1000):
        self.source_dir = source_dir
        super(ImageActivationGenerator, self).__init__(
            model, acts_dir, custom,max_examples)

    def get_examples_for_concept(self, concept):
        concept_dir = os.path.join(self.source_dir, concept)
        img_paths = [os.path.join(concept_dir, d)
                     for d in os.listdir(concept_dir)]
        imgs, filename_to_check = self.load_images_from_files(img_paths, 1000,
                                           shape=self.model.get_image_shape()[:2])
        self.data_filename = filename_to_check

        # TODO: print(imgs.shape)
        return imgs

    def load_image_from_file(self, filename, shape):
        """Given a filename, try to open the file. If failed, return None.

        Args:
            filename: location of the image file
            shape: the shape of the image file to be scaled

        Returns:
            the image if succeeds, None if fails.

        Rasies:
            exception if the image was not the right shape.
        """
        if not os.path.exists(filename):
            print('not exist')
        else:
            # ensure image has no transparency channel
            img = PIL.Image.open(filename)
            if self.custom:

                img1 = transform(img)
            else:
                img1 = transform_pretrained(img)

            if not (len(img1.shape) == 3 and img1.shape[0] == 3):
                return None
            else:
                return img1

    def load_images_from_files(self, filenames, max_imgs=1000,
                               do_shuffle=False, run_parallel=False,
                               shape=(128, 128),
                               num_workers=10):
        """Return image arrays from filenames.

        Args:
          filenames: locations of image files.
          max_imgs: maximum number of images from filenames.
          do_shuffle: before getting max_imgs files, shuffle the names or not
          run_parallel: get images in parallel or not
          shape: desired shape of the image
          num_workers: number of workers in parallelization.

        Returns:
          image arrays

        """
        
        imgs = []
        # print('length of filename',len(filenames))
        # First shuffle a copy of the filenames.
        filenames = filenames[:]
        # if do_shuffle:
        #     np.random.shuffle(filenames)

        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            imgs = pool.map(
                lambda filename: self.load_image_from_file(filename, shape),
                filenames[:max_imgs])
            imgs = [img for img in imgs if img is not None]
            if len(imgs) <= 1:
                raise ValueError('You must have more than 1 image in each class to run TCAV.')
        else:
            for filename in filenames:
                img = self.load_image_from_file(filename, shape)
                # print(img.size)
                if img is not None:
                    imgs.append(img)
                if len(imgs) < 1:
                    raise ValueError('You must have more than 1 image in each class to run TCAV.')
                elif len(imgs) >= max_imgs:
                    break
        return imgs, filenames