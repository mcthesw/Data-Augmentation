import cv2
import numpy
import json
from os import path


def get_mask(points: list, shape: tuple) -> numpy.ndarray:
    """

    :param points:
    :param shape:图片的长与宽
    :return: 使用ndarray储存的三通道uint8图片
    """
    blank_mask = numpy.zeros((shape[0], shape[1]))
    points = numpy.array(points, "int32")
    mask = cv2.fillConvexPoly(blank_mask, points, (255, 255, 255))
    return mask


def dump_mask(out_path: str, name, mask: numpy.ndarray):
    cv2.imwrite(path.join(out_path, name), mask)
    return


def get_image(file_path: str) -> numpy.ndarray:
    return cv2.imread(file_path)


def write_image(file_path: str, file_name: str, image: numpy.ndarray):
    cv2.imwrite(path.join(file_path, file_name + ".png"), image)


def read_masks_from_json(file_path: str) -> dict:
    """
    返回的dict结构如下
    {
        "n":[mask1,mask2,mask3,],
        "l":[mask4,mask5,mask6,],
        "h":[mask7,mask8,mask9,],
    }
    :param file_path: 给出的json文件路径
    :return: 一个包含了该json所有mask的字典，其中mask是ndarray
    """
    types = set()
    with open(file_path, mode="r") as file:
        json_file = json.loads(file.read())
    for i in json_file["shapes"]:
        types.add(i["label"][0])  # 只有第一个字母代表类型

    result = dict()
    for i in types:
        result[i] = []
    polygons = json_file["shapes"]
    shape = (json_file["imageHeight"], json_file["imageWidth"])
    for i in polygons:
        mask = get_mask(i["points"], shape)
        # i["label"][0]代表类型，可能是"n","l"之类的
        result[i["label"][0]].append(mask)
    return result
