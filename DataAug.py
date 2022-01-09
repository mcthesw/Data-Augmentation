import imgaug as ia
import imgaug.augmenters as iaa

import DataObj


def aug_data(data: DataObj.ImageData):
    aug = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=10),
        iaa.Fliplr(),  # 镜像对称
        iaa.Affine(translate_px={"x": (-40, 40)}, rotate=(-45, 45)),
    ])
    segmaps_aug = dict()
    for cur_type in data.masks.keys():
        cur_masks = [ia.Polygon(mask) for mask in data.masks[cur_type]]
        images_aug, tmp_segmaps_aug = aug(image=data.image, polygons=cur_masks)
        segmaps_aug[cur_type] = [i.coords for i in tmp_segmaps_aug]  # 把多边形转回坐标
    return DataObj.ImageData(data.name + "aug", images_aug, segmaps_aug)
