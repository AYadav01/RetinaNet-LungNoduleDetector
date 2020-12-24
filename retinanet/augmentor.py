import numpy as np
from skimage.util import random_noise
import torch

class VerticalFlip:

    def __call__(self, sample, flip_y=0.5):
        if np.random.rand() < flip_y:
            image, annots = sample['img'], sample['annot']
            image = np.flipud(image).astype(float)
            rows, cols, channels = image.shape
            # Get the y1 and y2 coords
            x1 = annots[:, 1].copy()
            x2 = annots[:, 3].copy()
            # Subtract from rows to get new coords
            x_tmp = x1.copy()
            annots[:, 1] = rows - x2
            annots[:, 3] = rows - x_tmp
            sample = {'img': image, 'annot': annots}
            return sample
        else:
            return sample

class HorizontalFlip:

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            # image = image[:, ::-1, :] #Horizontal Flip
            image = np.fliplr(image).astype(float)
            rows, cols, channels = image.shape
            # Get the x1 and x2 coords
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            # Subtract from cols to get new coords
            x_tmp = x1.copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
            sample = {'img': image, 'annot': annots}
            return sample
        else:
            return sample

class RandomNoise:
    def __call__(self, data, prob=0.2):
        if np.random.rand() < prob:
            img, annot = data['img'], data['annot']
            noised_img = random_noise(img).astype(float)
            return {"img": noised_img, "annot": annot}
        else:
            return data

class Normalizer:

    def __init__(self):
        # For Mayo Test Set - Full dose
        # self.mean = np.array([[[0.1288, 0.1288, 0.1288]]])
        # self.std = np.array([[[0.1648, 0.1648, 0.1648]]])

        # For Mayo Test Set - Low dose
        self.mean = np.array([[[0.1246, 0.1246, 0.1246]]])
        self.std = np.array([[[0.1551, 0.1551, 0.1551]]])

        # For LIDC
        # self.mean = np.array([[[0.2642, 0.3222, 0.4037]]])
        # self.std = np.array([[[0.1103, 0.3319, 0.0901]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer:

    def __init__(self, mean=None, std=None):
        if mean is None:
            # self.mean = [0.1288, 0.1288, 0.1288] # Full dose
            self.mean = [0.1246, 0.1246, 0.1246] # Low dose
        else:
            self.mean = mean
        if std is None:
            # self.std = [0.1648, 0.1648, 0.1648] # Full dose
            self.std = [0.1551, 0.1551, 0.1551] # Low dose
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class ToTensor:

    def __call__(self, data):
        img, annot = data['img'], data['annot']
        # Change channel dimension
        img = img.transpose((2, 0, 1))
        tensored_img, tensored_annot = torch.from_numpy(img), torch.from_numpy(annot)
        return {"img": tensored_img, "annot": tensored_annot}