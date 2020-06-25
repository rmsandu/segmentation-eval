import numpy as np
import nibabel as nib


def image_to_np(image):
    image_np = np.asanyarray(image.dataobj)
    if np.min(image_np) != np.max(image_np):
        image_np = image_np == np.max(image_np)
    return image_np


def load_image(file):
    image = nib.load(file)
    image = nib.as_closest_canonical(image)
    # print(nib.aff2axcodes(image.affine))
    image_np = image_to_np(image)

    return image, image_np
