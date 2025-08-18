import numpy as np
from math import exp
import torch
import torchvision.transforms.functional as TF

class AugmentationPipeline:
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict ={
            'translation': hparams.aug_weight_translation,
            'rotation': hparams.aug_weight_rotation,
            'scaling': hparams.aug_weight_scaling,
            'shearing': hparams.aug_weight_shearing,
            'rot90': hparams.aug_weight_rot90,
            'fliph': hparams.aug_weight_fliph,
            'flipv': hparams.aug_weight_flipv
        }
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState(hparams.seed)
    
    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else: 
            return False
        
    def set_augmentation_strength(self, p):
        self.augmentation_strength = p
    
    def _get_affine_padding_size(self, image, angle, scale, shear):
        h, w = image.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.], 
            [h/2, w/2, 1.], 
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1) 
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    def _get_translate_padding_and_crop(self, image, translation):
        t_x, t_y = translation
        h, w = image.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1)
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1)
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1)
            left = 0
        else:
            pad[2] = min(-t_y, w - 1)
            left = pad[2]
        return pad, top, left

    def augment_image(self, image):
        image = image.permute(3, 0, 1, 2)
        auged = False
        
        if self.random_apply('fliph'):
            auged = True
            image = TF.hflip(image)

        if self.random_apply('flipv'):
            auged = True
            image = TF.vflip(image)

        if self.random_apply('rot90'):
            auged = True
            k = self.rng.randint(1, 4)  
            image = torch.rot90(image, k, dims=[-2, -1])

        if self.random_apply('translation'):
            auged = True
            h, w = image.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)
            
            pad, top, left = self._get_translate_padding_and_crop(image, (t_x, t_y))
            image = TF.pad(image, padding=pad, padding_mode='reflect')
            image = TF.crop(image, top, left, h, w)

        interp = False 

        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1-self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        if interp:
            h, w = image.shape[-2:]
            pad = self._get_affine_padding_size(image, rot, scale, (shear_x, shear_y))
            image = TF.pad(image, padding=pad, padding_mode='reflect')
            image = TF.affine(
                image,
                angle=rot,
                scale=scale,
                shear=(shear_x, shear_y),
                translate=[0, 0],
                interpolation=TF.InterpolationMode.BILINEAR
            )
            image = TF.center_crop(image, (h, w))
        
        image = image.permute(1, 2, 3, 0)
        return image, auged or interp

class DataAugmentor:
    def __init__(self, hparams):
        self.hparams = hparams
        self.aug_on = hparams.aug_on
        self.epoch = 0
        if self.aug_on:
            self.augmentation_pipeline = AugmentationPipeline(hparams)
    
    def schedule_p(self):
        D = self.hparams.aug_delay
        T = self.hparams.num_epochs
        t = self.epoch
        p_max = self.hparams.aug_strength

        if t <= D:
            return 0.0
        else:
            c = self.hparams.aug_exp_decay/(T-D)
            p = p_max/(1-exp(-(T-D)*c))*(1-exp(-(t-D)*c))
        return p

    def __call__(self, image):
        if self.aug_on:
            p = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0

        is_aug = False
        if self.aug_on and p > 0.0:
            image, auged = self.augmentation_pipeline.augment_image(image)
            if auged: is_aug = True

        return image, is_aug