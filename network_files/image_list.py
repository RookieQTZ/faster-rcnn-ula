from typing import List, Tuple
from torch import Tensor


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, ul_tensors, image_sizes):
        # type: (Tensor, Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor) padding后的图像数据
            ul_tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
        """
        self.tensors = tensors
        self.ul_tensors = ul_tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        cast_ul_tensor = self.ul_tensors.to(device)
        return ImageList(cast_tensor, cast_ul_tensor, self.image_sizes)

