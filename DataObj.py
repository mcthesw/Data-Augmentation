from os import path
import os
import numpy

from Utils import dump_mask, get_image, read_masks_from_json, write_image


class ImageData:
    @classmethod
    def create_from_file(cls, file_name: str, source_path: str):
        """通过文件名和路径来获取数据，需要图片和同名json"""
        file_path = path.join(source_path, file_name)  # 该文件的完整路径
        json_file = path.join(source_path, file_name[:-4] + ".json")
        image = get_image(file_path)
        masks = read_masks_from_json(json_file)
        return cls(file_name[:-4], image, masks)

    def __init__(self, file_name: str, image: numpy.ndarray, masks: dict):
        self.name = file_name  # 这个是用于保存的ID
        self.image = image
        self.masks = masks

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @property
    def types(self) -> set:
        types = set()
        for i in self.masks.keys():
            types.add(i)
        return types

    def dump_masks_and_image(self, target_path: str):
        """把图片和mask按照格式导出到target_path"""
        for mask_type in self.types:
            # 把mask中的各个类别分别输出
            folder_name = f"[{mask_type}]" + self.name
            mask_folder_path = path.join(target_path, folder_name, "masks")
            os.makedirs(mask_folder_path)
            for index in range(len(self.masks[mask_type])):
                # 导出mask文件
                cur_mask = self.masks[mask_type][index]
                cur_mask_name = str(index)
                dump_mask(mask_folder_path, cur_mask_name, cur_mask)
            # 导出对应图片
            image_name = self.name + ".png"
            image_path = path.join(target_path, folder_name, "images", self.name)
            os.makedirs(image_path)
            write_image(image_path, self.name, self.image)

    def __str__(self) -> str:
        describe = f"Name:{self.name} Shape:{self.shape} Types:{self.types}"
        return describe
