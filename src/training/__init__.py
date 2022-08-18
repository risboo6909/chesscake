import cv2 as cv
import glob
import os


def _get_label(file_path):
    file_name = os.path.basename(file_path)
    # classic_knight_black_1.png -> knight_black_1
    try:
        return file_name.split(".")[0].split("_", 1)[1].rsplit("_", 1)[0]
    except:
        pass


def load_images(train_folder_path, smooth=True, grayscale=True, crop_margin=4, resize=None):
    """Load images from folder and return them as numpy array"""

    images = []
    labels = []

    for file_path in sorted(glob.iglob(os.path.join(train_folder_path, "*.png"))):
        label = _get_label(file_path)

        if not label:
            continue

        for _ in range(10):
            img = cv.imread(file_path)
            if crop_margin > 0:
                img = img[crop_margin:-crop_margin, crop_margin:-crop_margin]
            if smooth:
                img = cv.bilateralFilter(img, 25, 75, 75)
            if grayscale:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if resize:
                img = cv.resize(img, resize)
            images.append(img)
            labels.append(label)

    # cv.imshow("Debug", images[355])
    # cv.waitKey(0)

    return images, labels
