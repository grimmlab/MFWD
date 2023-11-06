from pathlib import Path
from skimage import io as skio
from skimage.transform import resize
import pandas as pd
import numpy as np


def coords_to_relative(xmin, ymin, xmax, ymax, shp):
    return xmin/shp[0], ymin/shp[1], xmax/shp[0], ymax/shp[1]


def coords_to_absolute(xmin, ymin, xmax, ymax, shp):
    return int(xmin*shp[0]), int(ymin*shp[1]), int(xmax*shp[0]), int(ymax*shp[1])


def resize_bounding_boxes(df, old_shp, new_shp):
    for idx, val in df.iterrows():
        xmin, ymin, xmax, ymax = coords_to_relative(
            val.xmin, val.ymin, val.xmax, val.ymax, old_shp)
        xmin, ymin, xmax, ymax = coords_to_absolute(
            xmin, ymin, xmax, ymax, new_shp)
        df.at[idx, "xmin"] = xmin
        df.at[idx, "xmax"] = xmax
        df.at[idx, "ymin"] = ymin
        df.at[idx, "ymax"] = ymax
    return


def resize_mask(msk, shp):
    resized_mask = np.zeros(shp, dtype=np.uint8)
    for idx in np.unique(msk):
        if idx == 0:
            continue
        a = msk == idx  # get binary map of one instance
        msk_resized = resize_image(a, shp, 0, b_mask=True)
        instance = msk_resized == 1
        np.add(resized_mask, instance*idx, out=resized_mask)
    return resized_mask


def resize_image(img, shp: tuple, order: int, b_mask: bool):
    """
    :param b_mask: whether to resize an image or a mask
    :param img: numpy array to resize
    :param shp: tuple of the resulting shape
    :param order: order of interpolation
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return: resized numpy array
    """
    if b_mask:
        return resize(img, shp)
    else:
        return resize(img, shp, anti_aliasing=True, clip=True, preserve_range=True, order=order)


def save_resized_image(image, save_path: Path, filename: str, suffix: str):
    skio.imsave(f"{save_path}/{filename}.{suffix}",
                image, check_contrast=False)


if __name__ == "__main__":
    data_path = Path("/data/MFWD/")
    b_masks = True  # whether masks are present
    csv_path = data_path / "gt.csv"
    jpegs_path = data_path / "jpegs"
    new_shape = (514, 614)
    save_path = data_path.parent / f"resized_{new_shape}"
    save_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    original_shape = (2056, 2454)
    resize_bounding_boxes(df, original_shape, new_shape)
    df.to_csv(f"{save_path}/gt.csv", index=False)  # save the csv file
    for image_p in jpegs_path.glob("*/*/*.jpeg"):
        save_img_path = save_path / "jpegs" / \
            image_p.parent.parent.stem / image_p.parent.stem
        save_img_path.mkdir(exist_ok=True, parents=True)
        img = skio.imread(image_p)
        image_resized = resize_image(img, new_shape, 0, False)
        save_resized_image(image_resized, save_img_path,
                           image_p.stem, suffix="jpeg")
        if b_masks:
            save_msk_pan_path = save_path / "masks" / "panoptic_segmentation" / \
                image_p.parent.parent.stem / image_p.parent.stem
            save_msk_sem_path = save_path / "masks" / "semantic_segmentation" / \
                image_p.parent.parent.stem / image_p.parent.stem
            save_msk_pan_path.mkdir(exist_ok=True, parents=True)
            save_msk_sem_path.mkdir(exist_ok=True, parents=True)
            pan_ann_p = str(image_p).replace(
                "jpegs", "masks/panoptic_segmentation").replace("jpeg", "png")
            msk = skio.imread(pan_ann_p, as_gray=True)
            msk_resized = resize_mask(msk, new_shape)
            # semantic mask, if all instances are of the same class
            msk_resized_sem = msk_resized > 1
            msk_resized_sem = msk_resized_sem.astype(np.uint8)*255
            save_resized_image(msk_resized, save_msk_pan_path,
                               image_p.stem, suffix="png")
            save_resized_image(
                msk_resized_sem, save_msk_sem_path, image_p.stem, suffix="png")
