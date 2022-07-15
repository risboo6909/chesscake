import numpy as np
import cv2 as cv
import chess
import operator

from collections import defaultdict, Counter

# small letters are for black pieces
# capital letters are for white pieces
# all sorted alphabetically
labels = ["b", "B", "k", "K", "n", "N", "p", "P", "q", "Q", "r", "R"]


def recognize_pieces(models, cropped_squares, turn: str, bottom_left: str) -> str:
    flatten = []

    if turn == "White":
        turn = chess.WHITE
    else:
        turn = chess.BLACK

    if bottom_left == "A1":
        bottom_left = chess.A1
    elif bottom_left == "A8":
        bottom_left = chess.A8
    elif bottom_left == "H1":
        bottom_left = chess.H1
    elif bottom_left == "H8":
        bottom_left = chess.H8

    board = chess.Board()
    board.clear_board()

    for img in cropped_squares:
        img = cv.resize(img, (20, 20), interpolation=cv.INTER_LANCZOS4)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        flatten.append(img.flatten())

    # normalize pixels intensity
    flatten = np.array(flatten) / 255.0

    results = defaultdict(Counter)
    for mlp in models:
        for square_idx, res in zip(range(64), mlp.predict(flatten)):
            try:
                class_idx = list(res).index(1)
                results[square_idx][class_idx] += 1
            except ValueError:
                pass

    consensus = 1
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

    return board
