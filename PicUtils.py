import cv2
import numpy
from os import path


def get_mask(points: list, shape: tuple) -> numpy.ndarray:
    blankMask = numpy.zeros((shape[0], shape[1]))
    points = numpy.array(points, "int32")
    mask = cv2.fillConvexPoly(blankMask, points, (255, 255, 255))
    return mask


def dump_mask(out_path: str, name, mask: numpy.ndarray):
    cv2.imwrite(path.join(out_path, name), mask)
    return
