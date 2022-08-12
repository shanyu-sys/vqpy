"""
This is a demo VQPy implementation listing and storing all red moving vehicles to a json file.
"""

from typing import Optional

import numpy as np

import vqpy

class Vehicle(vqpy.VObjBase):
    """The class of vehicles"""
    required_fields = ['class_id', 'tlbr']
    @vqpy.property()
    @vqpy.postproc({'majority': 100})
    def license_plate(self):
        """The license plate of the vehicle"""
        return self.infer('license_plate', {'license_plate': 'openalpr'})

class ListMovingVehicle(vqpy.QueryBase):
    """The class obtaining all moving vehicles"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Vehicle,
                       'bbox_velocity': lambda x: x >= 0.1}
        select_cons = {'track_id': None,
                       'license_plate': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='moving')

# add general built-in function to library
@vqpy.vqpy_func_logger(['image'], ['dominant_color'], [], required_length=1)
def extract_color(obj, image: Optional[np.ndarray]):
    """scratchy implementation obtaining the dominant color from an image"""
    if image is None:
        return [None]
    datas = {}
    nrows, ncols, _ = image.shape
    for row in range(nrows):
        for col in range(ncols):
            v = np.round(image[row, col] * 255.0) / 32          # 8x8x8 color grid
            v = tuple(int(x) for x in v)
            if v not in datas:
                datas[v] = 1
            else: datas[v] += 1
    color, best = None, 0
    for new_color, value in datas.items():
        if value > best:
            color = np.array(new_color) * 32 / 255.0
            best = value
    return [color]

class ListRedMovingVehicle(ListMovingVehicle):
    """The class obtaining all red moving vehicles"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'dominant_color':
                        lambda x: np.dot(x / np.linalg.norm(x), np.array([1, 0, 0])) > 0.5}
        return vqpy.VObjConstraint(filter_cons=filter_cons, filename="redmoving")

vqpy.launch(cls_name=vqpy.COCO_CLASSES,
            cls_type={"car": Vehicle, "truck": Vehicle},
            workers=[ListRedMovingVehicle()])
