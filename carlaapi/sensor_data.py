"""sensor_data.py"""

from typing import Tuple
import carla
import numpy


def process_rgb_data(data: carla.Image) -> numpy.ndarray:
    image = numpy.copy(numpy.frombuffer(data.raw_data, dtype=numpy.uint8))
    image = numpy.reshape(image, (data.height, data.width, 4))[..., :3]

    return image


def process_optical_flow_data(data: carla.Image) -> numpy.ndarray:
    image = numpy.copy(numpy.frombuffer(data.get_color_coded_flow().raw_data, dtype=numpy.uint8))
    image = numpy.reshape(image, (data.height, data.width, 4))[..., :3]

    return image

def process_depth_data(data: carla.Image) -> numpy.ndarray:
    image = numpy.ndarray(shape=(data.height, data.width, 4), dtype=numpy.uint8, buffer=data.raw_data)
    image = (numpy.float32(image[:, :, 2]) + numpy.float32(image[:, :, 1]) * 256. + numpy.float32(image[:, :, 0]) * 256. * 256.) / (256. * 256. * 256. - 1.) * 1000.

    return image

def process_semantic_segmentation_data(data: carla.Image) -> numpy.ndarray:
    image = numpy.copy(numpy.frombuffer(data.raw_data, dtype=numpy.uint8))
    image = numpy.reshape(image, (data.height, data.width, 4))[..., 2]

    return image

def process_lidar_ray_cast_data(data: carla.LidarMeasurement) -> numpy.ndarray:
    points = numpy.copy(numpy.frombuffer(data.raw_data, dtype=numpy.float32))
    points = numpy.reshape(points, (-1, 4))[:, 0:3]
    points[:, 1] *= -1.

    return points

SemanticLidar_T= Tuple[Tuple[numpy.ndarray, numpy.ndarray], carla.SemanticLidarMeasurement]
def process_lidar_ray_cast_semantic_data(data: carla.SemanticLidarMeasurement) -> SemanticLidar_T:
    raw_points: numpy.ndarray = numpy.copy(numpy.frombuffer(
        data.raw_data, 
        dtype=numpy.dtype([('x', numpy.float32), ('y', numpy.float32), ('z', numpy.float32), ('CosAngle', numpy.float32), ('ObjIdx', numpy.uint32), ('ObjTag', numpy.uint32)])
    ))
    points: numpy.ndarray = numpy.stack([raw_points['x'], raw_points['y'], raw_points['z']], axis=1)
    points[:, 1] *= -1.
    object_tag: numpy.ndarray = numpy.uint8(raw_points['ObjTag'])

    return ((points, object_tag), data)
