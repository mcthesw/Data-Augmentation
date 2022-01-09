import imgaug.augmenters as iaa
import numpy as np
import DataObj


def aug_data(data: DataObj.ImageData):
    aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=10),
        iaa.Fliplr(),  # 镜像对称
        iaa.Affine(translate_px={"x": (-40, 40)}, rotate=(-45, 45)),
    ])
    images_aug, segmaps_aug = aug(image=data.image,segmentation_maps=data.masks["l"])
    return DataObj.ImageData("ass",images_aug,segmaps_aug)