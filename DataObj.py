import os
import pickle
from os import path
from random import randint
from typing import List

import numpy

from Utils import dump_mask, get_image, read_masks_from_json, write_image, get_mask, counter

patch_counter = counter()


class ImageData:
    @classmethod
    def create_from_file(cls, file_name: str, source_path: str):
        """通过文件名和路径来获取数据，需要图片和同名json
        :rtype: object
        """
        file_path = path.join(source_path, file_name)  # 该文件的完整路径
        json_file = path.join(source_path, file_name[:-4] + ".json")
        image = get_image(file_path)
        masks = read_masks_from_json(json_file)
        return cls(file_name[:-4], image, masks)

    def __init__(self, file_name: str, image: numpy.ndarray, mask_polygons: dict):
        # noinspection PyTypeChecker
        self.mask_images: dict = None
        self.name = file_name  # 这个是用于保存的ID
        self.image = image
        self.mask_polygons = mask_polygons
        self.shape: tuple = self.image.shape
        if mask_polygons is not None:
            self.convert_polygons_to_images()

    @property
    def types(self) -> set:
        types = set()
        if self.mask_polygons is not None:
            for i in self.mask_polygons.keys():
                types.add(i)
        else:
            for i in self.mask_images.keys():
                types.add(i)
        return types

    def convert_polygons_to_images(self):
        self.mask_images = dict()
        for mask_type in self.types:
            cur_masks = []
            for mask_polygon in self.mask_polygons[mask_type]:
                cur_masks.append(get_mask(mask_polygon, self.shape))
            self.mask_images[mask_type] = cur_masks

    def dump_masks_and_image(self, target_path: str):
        """把图片和mask按照格式导出到target_path"""
        assert self.mask_images is not None
        for mask_type in self.types:
            if not self.mask_images[mask_type]:
                print(f"文件 [{mask_type}]{self.name} 导出失败，原因是没有mask")
                # 如果无mask，则不导出
                continue
            # 把mask中的各个类别分别输出
            folder_name = f"[{mask_type}]" + self.name
            mask_folder_path = path.join(target_path, folder_name, "masks")
            os.makedirs(mask_folder_path)
            for index in range(len(self.mask_images[mask_type])):
                # 导出mask文件
                cur_mask = self.mask_images[mask_type][index]
                cur_mask_name = str(index)
                dump_mask(mask_folder_path, cur_mask_name, cur_mask)
            # 导出对应图片
            image_path = path.join(target_path, folder_name, "images")
            os.makedirs(image_path)
            write_image(image_path, f"[{mask_type}]" + self.name, self.image)

    def drop_empty_masks(self):
        """去掉空的mask"""
        for mask_type in self.types:
            self.mask_images[mask_type] = [mask for mask in self.mask_images[mask_type] if not (mask == 0).all()]

    def split(self, size: tuple):
        # 因为生成patch时需要mask，所以不能为空
        assert self.mask_images is not None
        results = []
        # 切开原始图像
        patch_images = split_img(self.image, size)
        # 把原始mask图片也切开
        patch_masks = dict()
        for mask_type in self.types:
            patch_masks[mask_type] = list()
            for origin_mask_image in self.mask_images[mask_type]:
                patch_masks[mask_type].append(split_mask(origin_mask_image, size))
        # 按顺序进行组合，放入Patch对象
        cnt = 1
        for i in range(len(patch_images)):
            cur_patch_image = patch_images[i]
            cur_patch_masks = dict()
            for mask_type in self.types:
                cur_patch_masks[mask_type] = list()
                for mask in patch_masks[mask_type]:
                    cur_patch_masks[mask_type].append(mask[i])
            # noinspection PyTypeChecker
            new_image_data = ImageData(self.name + f"_split[{cnt}]", cur_patch_image, None)
            new_image_data.mask_images = cur_patch_masks
            new_image_data.drop_empty_masks()
            results.append(new_image_data)
            cnt += 1
        return results

    def __str__(self) -> str:
        describe = f"Name:{self.name} Shape:{self.shape} Types:{self.types}"
        return describe


class Patch:

    @classmethod
    def create_from_image_data(cls, data: ImageData, patch_size: tuple = (128, 128)) -> list:
        # 因为生成patch时需要mask，所以不能为空
        assert data.mask_images is not None
        results = []
        # 切开原始图像
        patch_images = split_img(data.image, patch_size)
        # 把原始mask图片也切开
        patch_masks = dict()
        for mask_type in data.types:
            patch_masks[mask_type] = list()
            for origin_mask_image in data.mask_images[mask_type]:
                new_masks = split_mask(origin_mask_image, patch_size)
                patch_masks[mask_type].append(new_masks)
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

    @classmethod
    def load_from_folder(cls, source_path: str):
        result: List[cls] = []
        file_names = os.listdir(source_path)
        for name in file_names:
            with open(path.join(source_path, name)) as file:
                result.append(pickle.load(file))
        return result

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

    def save_to_file(self, target_path: str):
        with open(target_path, mode="w") as file:
            pickle.dump(self, file)

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
        assert data.mask_images is not None
        # 需要使用copy来解决引用问题
        new_data = ImageData(data.name + f"_patch[{next(patch_counter)}]", data.image.copy(), None)
        new_data.mask_images = data.mask_images.copy()
        if pos is None:
            # 如果没指定位置，则随机取点，取的点要保证能放下一个patch
            pos = (randint(0, new_data.shape[0] - self.shape[0]), randint(0, new_data.shape[1] - self.shape[1]))
        # 把patch的图片覆盖到原图指定位置上
        new_data.image[pos[0]:pos[0] + self.shape[0], pos[1]:pos[1] + self.shape[1], :] = self.image
        new_data.convert_polygons_to_images()
        # 把mask从小的变换到大坐标系中
        new_mask_images = dict()
        empty_mask = numpy.zeros((data.image.shape[0], data.image.shape[1]), dtype="uint8")
        for mask_type in self.mask_images.keys():
            new_mask_images[mask_type] = []
            for mask_index in range(len(self.mask_images[mask_type])):
                cur_big_mask = empty_mask.copy()
                cur_big_mask[pos[0]:pos[0] + self.shape[0], pos[1]:pos[1] + self.shape[1]] = \
                    self.mask_images[mask_type][mask_index]
                new_mask_images[mask_type].append(cur_big_mask)
        # 把mask也贴到原图上
        for mask_type in new_mask_images.keys():
            if mask_type not in new_data.mask_images.keys():
                new_data.mask_images[mask_type] = new_mask_images[mask_type]
            else:
                new_data.mask_images[mask_type] += new_mask_images[mask_type]
        return new_data

    def drop_empty_masks(self):
        """去掉空的mask"""
        # TODO: 检查下面的语句是否正确
        empty = numpy.zeros(self.shape, dtype="uint8")
        for mask_type in self.types:
            self.mask_images[mask_type] = [mask for mask in self.mask_images[mask_type] if not (mask == empty).all()]


def split_img(image: numpy.ndarray, size: tuple) -> list:
    patches = []
    for x in range(0, image.shape[0], size[0]):
        for y in range(0, image.shape[1], size[1]):
            patches.append(image[
                           x:x + size[0],
                           y:y + size[1],
                           :
                           ])
    return patches


def split_mask(mask: numpy.ndarray, size: tuple) -> list:
    patches = []
    for x in range(0, mask.shape[0], size[0]):
        for y in range(0, mask.shape[1], size[1]):
            patches.append(mask[
                           x:x + size[0],
                           y:y + size[1]
                           ])
    return patches
