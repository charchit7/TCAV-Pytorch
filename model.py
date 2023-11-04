import os
import logging
import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import gc
import os
import sys
from pathlib import Path
from tcav_utils import get_grads_key
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
from net import ArnieClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_path = os.path.join(Path(__file__).parent.absolute(), 'labels.txt')

class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass
        
    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.tensor(acts).to(device)
        inputs.requires_grad = True

        cutted_model = self.get_cutted_model(bottleneck_name).to(device)
        cutted_model.eval()
        outputs = cutted_model(inputs)
        outputs = outputs[:, y[0]]

        grad_outputs = torch.ones_like(outputs)
        grads = -torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs)[0]
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        gc.collect()
        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass
    
    @abstractmethod
    def id_to_label(self, id):
        pass

    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)
        acts_list = []
        self.model.to(device)
        
        for i in range(0, len(examples), 16):
            batch_examples = examples[i:i + 16]
        
            inputs = torch.stack(batch_examples).to(device)
            with torch.no_grad():
                self.model.eval()
                self.model(inputs)

            acts_batch = bn_activation.detach().cpu().numpy()
            acts_list.append(acts_batch)

        acts = np.concatenate(acts_list, axis=0)
        handle.remove()
        torch.cuda.empty_cache()

        return acts
    

class ImageModelWrapper(ModelWrapper):
    """Wrapper base class for image models."""

    def __init__(self, image_shape):
        super(ModelWrapper, self).__init__()
        # shape of the input image in this model
        self.image_shape = image_shape

    def get_image_shape(self):
        """returns the shape of an input image."""
        return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
    """Simple wrapper of the public image models with session object."""

    def __init__(self, image_shape):
        super(PublicImageModelWrapper, self).__init__(image_shape=image_shape)
        try:
            self.labels = dict(eval(open(labels_path).read()))
        except:
            self.labels = open(labels_path).read().splitlines()

    def label_to_id(self, label):
        if isinstance(self.labels, dict):
            return list(self.labels.keys())[list(self.labels.values()).index(label)]
        else:
            return self.labels.index(label)

    def id_to_label(self, id):
        if isinstance(self.labels, dict):
            return self.labels[id]
        else:
            return self.labels[id]


############################ ARNIE CUSTOM HOOKS ##########################
##########################################################################

class ArnieClassification_cutted(nn.Module):
    def __init__(self, arnie_model, bottleneck):
        super(ArnieClassification_cutted, self).__init__()
        names = list(arnie_model._modules.keys())
        layers = list(arnie_model._modules.values())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue # output present
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            # pre-forward process
            if self.layers_names[i] == 'fc':
                y = y.view(y.size(0), -1)

            y = layer(y)
        return y
    

class ArnieWrapper(PublicImageModelWrapper):

    def __init__(self):
        image_shape = [128, 128, 3]
        super(ArnieWrapper, self).__init__(image_shape=image_shape)
        self.model = ArnieClassification()
        pth = './data/save_arnie.pth'
        self.model.load_state_dict(torch.load(pth))
        self.model_name = 'arnie_model'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return ArnieClassification_cutted(self.model, bottleneck)

def get_model_wrapper(name):
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name)

def get_or_load_gradients(model, acts, grads_dir, target_class, bottleneck):
    if grads_dir is None:
        grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
    else:
        path = os.path.join(grads_dir, get_grads_key(target_class, model.model_name, bottleneck))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                grads = np.load(f, allow_pickle=False)
                logging.info(path + ' exists and loaded, shape={}.'.format(str(grads.shape)))
        else:
            grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
            with open(path, 'wb') as f:
                np.save(f, grads, allow_pickle=False)
                logging.info(path + ' created, shape={}.'.format(str(grads.shape)))
    return grads.reshape([grads.shape[0], -1])


########################## RESNET18 ######################################
##########################################################################

class resnet18_cutted(torch.nn.Module):
    def __init__(self, resnet, bottleneck):
        super(resnet18_cutted, self).__init__()
        names = list(resnet._modules.keys())
        layers = list(resnet._modules.values())

        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # already calculated
            if not bottleneck_met:
                continue
            if name == 'aux1':
                continue
            if name == 'aux2':
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            # pre-forward process
            if self.layers_names[i] == 'fc':
                y = y.view(y.size(0), -1)

            y = layer(y)
        return y
    

class ResNet18Wrapper(PublicImageModelWrapper):

    def __init__(self):
        image_shape = [256, 256, 3]
        super(ResNet18Wrapper, self).__init__(image_shape=image_shape)
        self.model = resnet18()
        self.model.fc = nn.Linear(512,2)
        pth = '/DATA/charchit.sharma/final_submission/TCAV_functionality/source_dir/arnie_model3.pth'
        self.model.load_state_dict(torch.load(pth))
        self.model_name = 'resnet18_public'

    def forward(self, x):
        return self.model.forward(x)

    def get_cutted_model(self, bottleneck):
        return resnet18_cutted(self.model, bottleneck)

def get_model_wrapper(name):
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name)

def get_or_load_gradients(model, acts, grads_dir, target_class, bottleneck):
    if grads_dir is None:
        grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
    else:
        path = os.path.join(grads_dir, get_grads_key(target_class, model.model_name, bottleneck))
        if os.path.exists(path):
            with open(path, 'rb') as f:
                grads = np.load(f, allow_pickle=False)
                logging.info(path + ' exists and loaded, shape={}.'.format(str(grads.shape)))
        else:
            grads = model.get_gradient(acts, [model.label_to_id(target_class)], bottleneck)
            with open(path, 'wb') as f:
                np.save(f, grads, allow_pickle=False)
                logging.info(path + ' created, shape={}.'.format(str(grads.shape)))
    return grads.reshape([grads.shape[0], -1])