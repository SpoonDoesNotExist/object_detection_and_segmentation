import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from config import TRAIN_PATH


class CloddDataset(Dataset):
    """
        A custom PyTorch Dataset class for loading images and masks from a COCO annotation file.
    """

    def __init__(self, annFile, catNms):
        """
        :param annFile: Path to the COCO annotation file.
        :param catNms: List of category names to load.
        """
        self.coco = COCO(annFile)

        self.catIds = self.coco.getCatIds(catNms=catNms)
        self.imgIds = self.coco.getImgIds(catIds=self.catIds)

    def __getitem__(self, index):
        """
        Returns a dictionary containing the image, mask, and file path for the specified index.

        :param index: Index of the image to retrieve.
        :return:
        """
        image, mask, file_path = self.pull_item(index)
        image = image[2:-2, :, :]
        mask = mask[2:-2, :]

        image = self.__downscale(image)
        mask = self.__downscale(mask)

        image, mask = self._random_augmentation(image, mask)

        return {
            'image': image.transpose(-1, 0, 1),
            'mask': np.expand_dims(mask, axis=0)
        }

    def __downscale(self, image):
        """
        Downsamples the given image by a factor of 2 using nearest-neighbor interpolation.

        :param image: The image to downsample.
        :return:
        """
        height, width = image.shape[:2]
        return cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)

    def _random_augmentation(self, image, mask):
        """
        Applies a random horizontal or vertical flip or rotation to the given image and mask.

        :param image: The image to augment.
        :param mask: The mask to augment.
        :return:
        """
        # Flip horizontally
        if np.random.randint(0, 2):
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Flip vertically
        if np.random.randint(0, 2):
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        return image, mask

    def __len__(self):
        """
        Returns the number of images in the dataset.

        :return: The number of images in the dataset.
        """
        return len(self.imgIds)

    def pull_item(self, index):
        """
        Loads the raw image and mask for the specified index from the annotation file.

        :param index:
        :return:
        """
        img_info = self.coco.loadImgs(self.imgIds[index])[0]

        file_path = TRAIN_PATH / img_info['file_name'].replace('images/9', 'images_base')
        raw_img = cv2.imread(str(file_path))

        annIds = self.coco.getAnnIds(
            imgIds=img_info['id'],
            catIds=self.catIds,
            iscrowd=None
        )
        anns = self.coco.loadAnns(annIds)
        mask = sum(map(lambda x: self.coco.annToMask(x) > 0, anns))

        return raw_img, mask, file_path
