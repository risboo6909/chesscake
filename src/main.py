import pathlib
import os
import sys
import chess
import joblib

from result import Err
from cv_board import recognize_board
from nn_pieces import recognize_pieces

parent_dir = pathlib.Path(__file__).parent.parent.resolve()
samples_dir = os.path.join(parent_dir, "samples")


def load_models():
    models = []
    path_to_models = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "mlp")
    for file_name in os.listdir(path_to_models):
        if file_name.endswith(".joblib"):
            models.append(joblib.load(os.path.join(path_to_models, file_name)))

    return models


if __name__ == "__main__":
    image_path = os.path.join(samples_dir, "borisbot.png")
    cropped_squares = recognize_board(image_path)

    if isinstance(cropped_squares, Err):
        sys.exit(1)

    print("recognizing pieces...")
    models = load_models()
    result = recognize_pieces(
        models, cropped_squares.value, turn=chess.WHITE, bottom_left=chess.A1
    )
