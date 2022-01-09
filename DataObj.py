from os import path
import os
import json
import PicUtils
import shutil


class SourceFile:
    def __init__(self, name: str, source_path: str, target_path: str):
        self.name = name[:-4]  # 这个是用于保存的ID
        self.path = path.join(source_path, name)  # 该文件的完整路径
        self.target_path = target_path
        self.json_file = path.join(source_path, name[:-4] + ".json")
        """self.groups的结构应当如下,其中mask是一张可以被导出的mask二值图片\n
        (类型numpy.ndarray)\n
            {
                "n":[mask1,mask2,mask3,],
                "l":[mask4,mask5,mask6,],
                 .....
            }
        """

    @property
    def shape(self) -> tuple:
        with open(self.json_file, mode="r") as file:
            tmp = json.loads(file.read())
        return tmp["imageHeight"], tmp["imageWidth"]

    @property
    def types(self) -> set:
        types = set()
        with open(self.json_file, mode="r") as file:
            tmp = json.loads(file.read())
        for i in tmp["shapes"]:
            types.add(i["label"][0])  # 只有第一个字母代表类型
        return types

    @property
    def groups(self) -> dict:
        result = dict()
        for i in self.types:
            result[i] = []
        with open(self.json_file, mode="r") as file:
            tmp = json.loads(file.read())
        tmp = tmp["shapes"]
        for i in tmp:
            mask = PicUtils.get_mask(i["points"], self.shape)
            # i["label"][0]代表类型，可能是"n","l"之类的
            result[i["label"][0]].append(mask)
        return result

    def dump_masks(self):
        for type in self.types:
            # 创建对应文件夹
            folder_name = f"[{type}]" + self.name
            mask_folder_path = path.join(self.target_path, folder_name, "masks")
            os.makedirs(mask_folder_path)
            # 创建mask文件
            for index in range(len(self.groups[type])):
                mask = self.groups[type][index]
                file_name = str(index) + ".png"
                PicUtils.dump_mask(mask_folder_path, file_name, mask)
            # 复制源图片
            source_pic_path = path.join(self.target_path, folder_name, "images")
            os.makedirs(source_pic_path)
            target_path = path.join(source_pic_path, folder_name + ".png")
            shutil.copy(self.path, target_path)

    def __str__(self) -> str:
        describe = f"Name:{self.name} \nLocation:{self.path} \nShape:{self.shape} \nTypes:{self.types}"
        return describe
