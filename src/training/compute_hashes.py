import os
import cv2 as cv

from collections import defaultdict
from imagehash import whash as imghash
from PIL import Image
from src.training import load_images

if __name__ == "__main__":
    result = defaultdict(list)

    images, exp_outputs = load_images("src/training/data", smooth=True, grayscale=False)
    for image, exp_output in zip(images, exp_outputs):
        imghash(Image.fromarray(image))
        result[exp_output].append(imghash(Image.fromarray(image)))

    crop_margin = 4

    img = cv.imread(os.path.join("src/training/data/test2.png"))
    img = img[crop_margin:-crop_margin, crop_margin:-crop_margin]
    img = cv.bilateralFilter(img, 25, 75, 75)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    target_hash = imghash(Image.fromarray(img))

    min_val = 10000000
    min_label = ''

    for k, vals in result.items():
        for val in vals:
            if min_val > target_hash - val:
                min_val  = target_hash - val
                min_label = k

    print(min_label)