import numpy as np
import cv2 as cv
import chess
import operator

from mlp import labels
from result import Err, Ok
from collections import defaultdict, Counter


def recognize_pieces(
    models, cropped_squares, turn: chess.Color, bottom_left: chess.Square
):
    flatten = []

    board = chess.Board()
    board.clear_board()

    for img in cropped_squares:
        img = cv.resize(img, (20, 20), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flatten.append(gray.flatten())

    # normalize pixels intensity
    flatten = np.array(flatten) / 255.0

    results = defaultdict(Counter)
    for mlp in models:
        for square_idx, v in zip(range(64), mlp.predict_proba(flatten)):
            # get index of max value
            class_idx = np.argmax(v)
            # get most probable class
            if max(v) >= 0.5:
                results[square_idx][class_idx] += 1

    consensus = 3
    for square_idx, decisions in results.items():
        class_idx, agreed = max(decisions.items(), key=operator.itemgetter(1))
        if agreed >= consensus:
            board.set_piece_at(square_idx, chess.Piece.from_symbol(labels[class_idx]))

    if bottom_left == chess.A1:
        board = board.transform(chess.flip_vertical)
    elif bottom_left == chess.H1:
        board = board.transform(chess.flip_vertical)
        board = board.transform(chess.flip_horizontal)
    elif bottom_left == chess.H8:
        board = board.transform(chess.flip_horizontal)

    board.turn = turn

    print(board.unicode(empty_square=".", invert_color=True, borders=False))
    print(board.fen())
    # cv.imshow("Debug", cropped_squares[3])
    # cv.waitKey(0)
