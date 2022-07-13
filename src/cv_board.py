from collections import defaultdict
from result import Err, Ok
from points_area.points import PointsInArea
from sklearn.cluster import DBSCAN
from operator import itemgetter
from typing import List, Tuple
import cv2 as cv
import random
import numpy as np
import pygad
import sys

BOARD_SQUARES = 64


def draw_debug_lines(img, lines, color, size_x, size_y):
    """Draws debug lines"""

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + size_x * (-b)), int(y0 + size_y * (a)))
            pt2 = (int(x0 - size_x * (-b)), int(y0 - size_y * (a)))
            cv.line(img, pt1, pt2, color, 1, cv.LINE_AA)


def draw_segmented(img, segmented, intersections, recognized_squares, size_x, size_y):
    """Draws debug image"""

    for square in recognized_squares:
        cv.rectangle(
            img,
            (
                int(square[0]),
                int(square[1]),
            ),
            (
                int(square[2]),
                int(square[3]),
            ),
            random.choices(range(256), k=3),
            -1,
        )

    # if segmented:
    #     draw_debug_lines(img, segmented[0], (255, 0, 0), size_x, size_y)
    #     draw_debug_lines(img, segmented[1], (0, 255, 0), size_x, size_y)

    for isect in intersections:
        cv.rectangle(
            img,
            (int(isect[0]) - 1, int(isect[1]) + 1),
            (int(isect[0]) + 1, int(isect[1]) - 1),
            (255, 255, 255),
            2,
        )


def segment_by_angle_kmeans(lines, k=2):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.

    See https://stackoverflow.com/a/46572063
    """

    # define criteria = (type, max_iter, epsilon)
    attempts = 10
    criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.3

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])

    # multiply the angles by two and find coordinates of that angle
    pts = np.array(
        [[np.cos(2 * angle), np.sin(2 * angle)] for angle in angles], dtype=np.float32
    )

    # run kmeans on the coords
    labels, _ = cv.kmeans(pts, k, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)[
        1:
    ]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2, width, height):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """

    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])

    try:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        if x0 <= width and y0 <= height:
            return (
                x0,
                y0,
            )
    except:
        pass

    return None


def segmented_intersections(lines, width, height):
    """Finds the intersections between groups of lines"""
    intersections = []

    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1 :]:
            for line1 in group:
                for line2 in next_group:
                    intersect = intersection(line1, line2, width, height)
                    if intersect:
                        intersections.append(intersect)

    return intersections


def find_rectangles(sorted_intersections, deviation, min_side_len, max_side_len):
    """Finds rectangles between intersections of lines recognized on board.

    sorted_intersections # list of lines intersection points sorted by (x, y)
    deviation            # dispersion radius for corner points in pixels
    min_side_len         # minimum square side length in pixels
    max_side_len         # maximum square side length in pixels
    """

    if min_side_len >= max_side_len:
        return []

    squares_found = set()
    fuzzy_pts = PointsInArea(sorted_intersections)

    for i, top_left in enumerate(sorted_intersections[:-1]):

        for bottom_right in sorted_intersections[i + 1 :]:

            x0, y0 = top_left
            x1, y1 = bottom_right

            x_len = abs(x1 - x0)
            y_len = abs(y1 - y0)

            if x1 <= x0 or x_len < min_side_len or x_len > max_side_len:
                continue

            if y1 <= y0 or y_len < min_side_len or y_len > max_side_len:
                continue

            if fuzzy_pts.has_points(x0, y1, deviation) and fuzzy_pts.has_points(
                x1, y0, deviation
            ):
                # sqaure recognized
                squares_found.add((x0, y0, x1, y1))

    return squares_found


def group_by_rectangle_area(rectangles, area_delta: float):
    """Groups rectangles by area"""

    data = np.array([(abs(r[0] - r[2]), abs(r[1] - r[3])) for r in rectangles])
    db = DBSCAN(eps=area_delta / 100, min_samples=10, n_jobs=-1)
    db.fit(data)

    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        clusters[label].append(rectangles[i])

    return clusters


def rectangles_overlap(
    recognized_rectangles: List[Tuple[float, float, float, float]],
):
    """Checks if rectangles overlap"""

    num_overlaps = 0

    for i, r1 in enumerate(recognized_rectangles):
        l1_x, l1_y, r1_x, r1_y = r1

        for r2 in recognized_rectangles[i + 1 :]:
            l2_x, l2_y, r2_x, r2_y = r2

            # if one rectangle is on left side of other
            if l1_x >= r2_x or l2_x >= r1_x:
                continue

            # if one rectangle is above other
            if r1_y <= l2_y or r2_y <= l1_y:
                continue

            # intersection found
            num_overlaps += 1

    return num_overlaps


def thin_out_intersections(sorted_intersections, radius):
    """Thins out intersections of lines recognized on board."""

    res = []
    scanned_points = set()

    fuzzy_pts = PointsInArea(sorted_intersections)
    for p in sorted_intersections:
        if p in scanned_points:
            continue
        c_x, c_y, neighbours = fuzzy_pts.get_center_of_mass(p[0], p[1], radius)
        scanned_points.update(neighbours)
        res.append(
            (
                c_x,
                c_y,
            )
        )

    return list(sorted(res, key=itemgetter(0)))


def resize_image(img, size_x: int, size_y: int):
    resized = cv.resize(img, (size_x, size_y), interpolation=cv.INTER_CUBIC)
    return resized


def grayscale_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def blur(img, k_size_x: int, k_size_y: int):
    blurry = cv.GaussianBlur(img, (k_size_x, k_size_y), 0)
    return blurry


def canny_image(img, threshold1, threshold2):
    canny = cv.Canny(img, threshold1, threshold2, L2gradient=True)
    return canny


def rect_distances(rectangles, width, height):
    """Calculates how many outstanders between rectangle distances"""

    max_distance = 10
    outstanders = 0

    # compute rectangle center points
    center_points = [
        (
            (r[0] + r[2]) / 2,
            (r[1] + r[3]) / 2,
        )
        for r in rectangles
    ]

    for p in center_points:
        if p[0] < 0 or  p[0] > width:
            outstanders += 1
        elif p[1] < 0 or  p[1] > height:
            outstanders += 1

    if len(rectangles) < 64:
        return 0

    # try x axist first and then y axis
    for coord_idx, sort_order in enumerate([(1, 0), (0, 1)]):

        # sort centers from top left to bottom right
        center_points = sorted(center_points, key=itemgetter(*sort_order))

        # compute distances between center points
        distances = [
            abs(p1[coord_idx] - p2[coord_idx])
            for p1, p2 in zip(center_points, center_points[1:])
            if p1[coord_idx] < p2[coord_idx]
        ]

        distances = sorted(distances)

        # compute outstanding distances
        if len(distances) > 1:
            first_delta = distances[1] - distances[0]
            # find how many rectangles there with the distances between them more than max_distance
            for i, d in enumerate(zip(distances, distances[1:])):
                delta = d[1] - d[0]
                if delta - first_delta > max_distance:
                    outstanders += len(distances) - i - 1
                    break

    return outstanders


def compute_fitness(squares_num: int, invalid_num: int):
    """Computes fitness of recognized squares"""
    if squares_num <= BOARD_SQUARES:
        return squares_num - invalid_num

    return BOARD_SQUARES - (squares_num - BOARD_SQUARES) - invalid_num


def scan(img, inp, debug):
    (
        threshold1,
        threshold2,
        min_points_per_line,
        k_size_x,
        k_size_y,
        side_len,
        min_side_len,
        max_side_len,
        area_delta,
        max_lines,
        point_size,
    ) = inp

    img_height, img_width, _ = img.shape

    best_cluster = []
    squares_num = overlaps_num = 0
    best_fitness, smallest_invalid = -sys.maxsize, sys.maxsize

    if threshold2 <= threshold1:
        return best_fitness, best_cluster

    prepared = canny_image(
        blur(grayscale_image(img), k_size_x, k_size_y), threshold1, threshold2
    )

    # try to distinguish straight
    lines = cv.HoughLines(prepared, 1, np.pi / 180, min_points_per_line, None, 0, 0)
    if lines is None:
        return best_fitness, best_cluster

    lines = lines[:max_lines]

    # segmented will contian vertical and horizontal lines
    segmented = segment_by_angle_kmeans(lines)

    # find intersections of horizontal and vertical lines
    intersections = segmented_intersections(segmented, img_width, img_height)

    # thin out intersections
    sorted_intersections = thin_out_intersections(intersections, point_size)

    # find rectangles based on lines intersections
    recognized_rectangles = find_rectangles(
        sorted_intersections, side_len, min_side_len, max_side_len
    )

    if recognized_rectangles:

        clusters = group_by_rectangle_area(list(recognized_rectangles), area_delta)

        # find best cluster
        for cluster_id, cluster in clusters.items():
            if cluster_id == -1:
                continue

            overlaps_num = rectangles_overlap(cluster)
            outstanders_num = rect_distances(cluster, img_width, img_height)

            invalid_squares = overlaps_num + outstanders_num

            squares_num = compute_fitness(len(cluster), invalid_squares)
            if squares_num > best_fitness and smallest_invalid > invalid_squares:
                best_fitness = squares_num
                smallest_invalid = invalid_squares
                best_cluster = cluster

            if best_fitness == BOARD_SQUARES:
                break

    if debug and best_cluster:
        draw_segmented(
            img,
            segmented,
            sorted_intersections,
            best_cluster,
            img_width,
            img_height,
        )

    return best_fitness, best_cluster


def do_recognize(img):
    def inner(inp, _):
        best_fitness, _ = scan(img, inp, debug=False)
        return best_fitness

    return inner


def crop_squares(img, rectangles: List[Tuple[float, float, float, float]]):
    """Crops squares from image"""

    # sort rectangles from top left to bottom right
    rectangles = sorted(rectangles, key=itemgetter(1, 0))

    cropped = []
    for rect in rectangles:
        x0, y0, x1, y1 = rect
        cropped.append(img[int(y0) : int(y1), int(x0) : int(x1)])

    return cropped


def func_generation(ga_instance):
    print(ga_instance.best_solution())


def from_file_object(file):
    file.seek(0)
    img_array = np.asanyarray(bytearray(file.read()), dtype=np.uint8)
    img = cv.imdecode(img_array, cv.IMREAD_COLOR)
    return img


def from_path(file_path):
    return cv.imread(file_path)


def recognize_board(img):
    """Recognizes board from image"""

    # draw border which helps in case of board occupies the whole size of the picture for algorithm to be
    # able to detect board edges
    img = cv.copyMakeBorder(
        img, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=[255, 255, 255]
    )

    img_height, img_width, _ = img.shape

    board_found = False

    # dynamic parameters for genetic algorithm
    # rectangles_group_epsilon = 110
    # max_lines = 30
    # sol_per_pop = 50

    rectangles_group_epsilon = 300
    max_lines = 40
    sol_per_pop = 100

    # add size constraints?
    # img = imutils.resize(img, width=500, inter=cv.INTER_LANCZOS4)

    for _ in range(1):

        print("new epoch")

        find_board = do_recognize(img)

        ga_instance = pygad.GA(
            num_parents_mating=6,
            num_generations=12,
            mutation_type="adaptive",
            fitness_func=find_board,
            mutation_probability=[0.7, 0.2],
            mutation_num_genes=[5, 2],
            mutation_percent_genes=[50, 20],
            crossover_probability=0.2,
            on_generation=func_generation,
            gene_space=[
                range(1, 100),  # threshold1 for Canny
                range(1, 200),  # threshold2 for Canny
                range(
                    80, max(img_width, img_height) + 2
                ),  # minimum points laying on line
                range(1, 11, 2),  # gauss
                range(1, 11, 2),  # gauss
                range(0, 8),  # maximum dispersion for a point of rectangle
                range(10, 80, 3),  # minimum side length
                range(10, 300, 3),  # maximum side length
                range(30, rectangles_group_epsilon),  # epsilon for DBSCAN algorithm
                range(18, max_lines),  # total number of lines
                range(5, 30, 2),  # point size
            ],
            gene_type=int,
            sol_per_pop=sol_per_pop,
            num_genes=11,
            parallel_processing=["thread", 4],  # looks like this feature is buggy
            stop_criteria=["reach_64", "saturate_8"],
        )

        ga_instance.run()
        solution, fitness, _ = ga_instance.best_solution()

        if fitness == BOARD_SQUARES:
            board_found = True
            break

        # increase possible uncertainty for rectangle side sizes
        rectangles_group_epsilon += 40
        max_lines += 5
        sol_per_pop += 10

    if not board_found:
        return Err("Board not found")

    _, best_cluster = scan(img, solution, debug=False)
    cropped = crop_squares(img, best_cluster)

    # cv.imshow("Debug", img)
    # cv.waitKey(0)
    return Ok(cropped)
