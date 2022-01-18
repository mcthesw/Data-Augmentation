import os
import random
from os import path
from random import randint

import numpy

from Utils import dump_mask, get_image, read_masks_from_json, write_image, get_mask


class ImageData:
    @classmethod
    def create_from_file(cls, file_name: str, source_path: str):
        """通过文件名和路径来获取数据，需要图片和同名json"""
        file_path = path.join(source_path, file_name)  # 该文件的完整路径
        json_file = path.join(source_path, file_name[:-4] + ".json")
        image = get_image(file_path)
        masks = read_masks_from_json(json_file)
        return cls(file_name[:-4], image, masks)

    def __init__(self, file_name: str, image: numpy.ndarray, mask_polygons: dict):
        # noinspection PyTypeChecker
        self.mask_image: dict = None
        self.name = file_name  # 这个是用于保存的ID
        self.image = image
        self.mask_polygons = mask_polygons
        self.shape: tuple = self.image.shape

    @property
    def types(self) -> set:
        types = set()
        for i in self.mask_polygons.keys():
            types.add(i)
        return types

    def convert_polygons_to_images(self):
        self.mask_image = dict()
        for mask_type in self.types:
            cur_masks = []
            for mask_polygon in self.mask_polygons[mask_type]:
                cur_masks.append(get_mask(mask_polygon, self.shape))
            self.mask_image[mask_type] = cur_masks

    def dump_masks_and_image(self, target_path: str):
        """把图片和mask按照格式导出到target_path"""
        assert self.mask_image is not None
        for mask_type in self.types:
            # 把mask中的各个类别分别输出
            folder_name = f"[{mask_type}]" + self.name
            mask_folder_path = path.join(target_path, folder_name, "masks")
            os.makedirs(mask_folder_path)
            for index in range(len(self.mask_image[mask_type])):
                # 导出mask文件
                cur_mask = self.mask_image[mask_type][index]
                cur_mask_name = str(index)
                dump_mask(mask_folder_path, cur_mask_name, cur_mask)
            # 导出对应图片
            image_path = path.join(target_path, folder_name, "images", self.name)
            os.makedirs(image_path)
            write_image(image_path, self.name, self.image)

    def __str__(self) -> str:
        describe = f"Name:{self.name} Shape:{self.shape} Types:{self.types}"
        return describe


class Patch:

    @classmethod
    def create_from_image_data(cls, data: ImageData, patch_size: tuple = (128, 128)) -> list:
        # 因为生成patch时需要mask，所以不能为空
        assert data.mask_image is not None
        results = []
        # 切开原始图像
        patch_images = split_img(data.image, patch_size)
        # 把原始mask图片也切开
        patch_masks = dict()
        for mask_type in data.types:
            patch_masks[mask_type] = list()
            for origin_mask_image in data.mask_image[mask_type]:
                patch_masks[mask_type].append(split_mask(origin_mask_image, patch_size))
        # 按顺序进行组合，放入Patch对象
        for i in range(len(patch_images)):
            cur_patch_image = patch_images[i]
            cur_patch_masks = dict()
            for mask_type in data.types:
                cur_patch_masks[mask_type] = list()
                for mask in patch_masks[mask_type]:
                    cur_patch_masks[mask_type].append(mask[i])
            results.append(Patch(cur_patch_image, cur_patch_masks))
        return results

    def __init__(self, image: numpy.ndarray, mask_images: dict):
        self.image = image
        self.shape = image.shape[0:2]
        self.types = mask_images.keys()
        self.mask_images = mask_images
        self.drop_empty_masks()

    def check_include_target(self) -> bool:
        """检查是存在mask，存在则返回True"""
        if len(self.mask_images.values()) == 0:
            return False
        for i in self.mask_images.values():
            if len(i) > 0:
                return True
        else:
            return False

    def check_boundary(self) -> bool:
        """
        检查所有mask的边界是否为白色（有物体被分割）
        True表示正常，False表示有物体被分割
        """
        for mask_type in self.mask_images.keys():
            for mask_image in self.mask_images[mask_type]:
                x, y = mask_image.shape
                white = 255
                for cur_x in range(x):
                    if (mask_image[cur_x, y - 1] == white or  # Shape中的x和y是从1开始的，所以需要-1
                            mask_image[cur_x, 0] == white):
                        return False
                for cur_y in range(y):
                    if (mask_image[x - 1, cur_y] == white or
                            mask_image[0, cur_y] == white):
                        return False
        return True

    def apply_to_image_data(self, data: ImageData, pos: tuple = None) -> ImageData:
        # 因为需要把新的mask覆盖到旧的上面，所以旧的必须存在
        assert data.mask_image is not None
        # TODO:需要解决命名冲突
        # 需要使用copy来解决引用问题
        new_data = ImageData(data.name + "_patch" + str(random.randint(10000, 99999)), data.image.copy(),
                             data.mask_polygons.copy())
        if pos is None:
            # 如果没指定位置，则随机取点，取的点要保证能放下一个patch
            pos = (randint(0, new_data.shape[0] - self.shape[0]), randint(0, new_data.shape[1] - self.shape[1]))
        # 把patch的图片覆盖到原图指定位置上
        new_data.image[pos[0]:pos[0] + self.shape[0], pos[1]:pos[1] + self.shape[1], :] = self.image
        new_data.convert_polygons_to_images()
        # 把mask从小的变换到大坐标系中
        empty_mask = numpy.zeros((data.image.shape[0], data.image.shape[1]), dtype="uint8")
        for mask_type in self.mask_images.keys():
            for mask_index in range(len(self.mask_images[mask_type])):
                cur_big_mask = empty_mask.copy()
                cur_big_mask[pos[0]:pos[0] + self.shape[0], pos[1]:pos[1] + self.shape[1]] = \
                    self.mask_images[mask_type][mask_index]
                self.mask_images[mask_type][mask_index] = cur_big_mask
        # 把mask也贴到原图上
        for mask_type in self.mask_images.keys():
            if mask_type not in new_data.mask_image.keys():
                new_data.mask_image[mask_type] = self.mask_images[mask_type]
            else:
                new_data.mask_image[mask_type] += self.mask_images[mask_type]
        return new_data

    def drop_empty_masks(self):
        """去掉空的mask"""
        empty = numpy.zeros(self.shape, dtype="uint8")
        for mask_type in self.types:
            self.mask_images[mask_type] = [mask for mask in self.mask_images[mask_type] if not (mask == empty).all()]


def split_img(image: numpy.ndarray, patch_size: tuple) -> list:
    patches = []
    for x in range(0, image.shape[0], patch_size[0]):
        for y in range(0, image.shape[1], patch_size[1]):
            patches.append(image[
                           x:x + patch_size[0],
                           y:y + patch_size[1],
                           :
                           ])
    return patches


def split_mask(mask: numpy.ndarray, patch_size: tuple) -> list:
    patches = []
    for x in range(0, mask.shape[0], patch_size[0]):
        for y in range(0, mask.shape[1], patch_size[1]):
            patches.append(mask[
                           x:x + patch_size[0],
                           y:y + patch_size[1]
                           ])
    return patches
