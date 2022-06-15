import pathlib
import os
import sys

from result import Ok, Err
from cv_board import recognize_board

parent_dir = pathlib.Path(__file__).parent.parent.resolve()
samples_dir = os.path.join(parent_dir, "samples")

if __name__ == "__main__":
    image_path = os.path.join(samples_dir, "board8.png")
    res = recognize_board(image_path)

    if isinstance(res, Err):
        print(res)
        sys.exit(1)
