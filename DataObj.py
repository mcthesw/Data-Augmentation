import os
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
        self.mask_image = None
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
    def __init__(self, image: numpy.ndarray, mask_images: dict):
        self.image = image
        self.shape = image.shape
        self.mask_images = mask_images

    def check_include_target(self) -> bool:
        """检查是存在mask，存在则返回True"""
        if len(self.mask_images.values()) == 0:
            return False
        else:
            return True

    def check_boundary(self) -> bool:
        """
        检查所有mask的边界是否为白色（有物体被分割）
        True表示正常，False表示有物体被分割
        """
        for mask_type in self.mask_images.keys():
            for mask_image in self.mask_images[mask_type]:
                x, y, z = mask_image.shape
                white = tuple([255 for i in range(z)])
                for cur_x in range(x):
                    if (mask_image[cur_x, y] == white or
                            mask_image[cur_x, 0] == white):
                        return False
                for cur_y in range(y):
                    if (mask_image[x, cur_y] == white or
                            mask_image[0, cur_y] == white):
                        return False
        return True

    def apply_to_image_data(self, data: ImageData, pos: tuple = None):
        if pos is None:
            # 如果没指定位置，则随机取点，取的点要保证能放下一个patch
            pos = (randint(0, data.shape[0] - self.shape[0]), randint(0, data.shape[1] - self.shape[1]))
        # 把patch的图片覆盖到原图指定位置上
        data.image[pos[0]:pos[0] + self.shape[0], pos[1]:pos[1] + self.shape[1], :] = self.image
        # TODO:把mask也贴到原图上
