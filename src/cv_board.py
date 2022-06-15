from collections import defaultdict
from result import Err
from points_area.points import PointsInArea
from sklearn.cluster import DBSCAN
import cv2 as cv
import random
import numpy as np
import pygad
import sys

BOARD_SQUARES = 64


def draw_debug_lines(img, lines, color, size_x, size_y):
    # draw the lines
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
            (square[0], square[1]),
            (square[2], square[3]),
            random.choices(range(256), k=3),
            -1,
        )

    # square = recognized_squares[10]
    # cv.rectangle(
    #     img,
    #     (square[0], square[1]),
    #     (square[2], square[3]),
    #     random.choices(range(256), k=3),
    #     -1,
    # )

    if segmented:
        draw_debug_lines(img, segmented[0], (255, 0, 0), size_x, size_y)
        draw_debug_lines(img, segmented[1], (0, 255, 0), size_x, size_y)

    for isect in intersections:
        isect = isect[0]
        cv.rectangle(
            img,
            (isect[0] - 1, isect[1] + 1),
            (isect[0] + 1, isect[1] - 1),
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
    criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.5

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
            return [[x0, y0]]
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

    # sort intersections by (x, y) coordinates
    return list(sorted(intersections, key=lambda pair: pair[0]))


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

            x0, y0 = top_left[0]
            x1, y1 = bottom_right[0]

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


def group_by_rectangle_area(rectangles, epsilon):
    data = np.array([(abs(r[0] - r[2]), abs(r[1] - r[3])) for r in rectangles])
    db = DBSCAN(eps=epsilon / 100, min_samples=10, n_jobs=-1)
    db.fit(data)

    clusters = defaultdict(list)
    for i, label in enumerate(db.labels_):
        clusters[label].append(rectangles[i])

    return clusters


def rectangles_overlap(recognized_rectangles):
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


def resize_image(img, size_x, size_y):
    resized = cv.resize(img, (size_x, size_y), interpolation=cv.INTER_CUBIC)
    return resized


def grayscale_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def blur(img, k_size_x, k_size_y):
    blurry = cv.GaussianBlur(img, (k_size_x, k_size_y), 0)
    return blurry


def canny_image(img, threshold1, threshold2):
    canny = cv.Canny(img, threshold1, threshold2, L2gradient=True)
    return canny


def compute_fitness(squares_num, overlaps_num):
    if squares_num <= BOARD_SQUARES:
        return squares_num - overlaps_num

    return BOARD_SQUARES - (squares_num - BOARD_SQUARES) - overlaps_num


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
        epsilon,
        max_lines,
    ) = inp

    img_height, img_width, _ = img.shape

    if threshold2 <= threshold1:
        return 0

    prepared = canny_image(
        blur(grayscale_image(img), k_size_x, k_size_y), threshold1, threshold2
    )

    # try to distinguish straight
    lines = cv.HoughLines(prepared, 1, np.pi / 180, min_points_per_line, None, 0, 0)
    if lines is None:
        return 0

    lines = lines[:max_lines]

    # segmented will contian vertical and horizontal lines
    segmented = segment_by_angle_kmeans(lines)

    # find intersections of horizontal and vertical lines
    sorted_intersections = segmented_intersections(segmented, img_width, img_height)

    # find rectangles based on lines intersections
    recognized_rectangles = find_rectangles(
        sorted_intersections, side_len, min_side_len, max_side_len
    )

    best_cluster = None
    squares_num = overlaps_num = 0
    best_fitness, smallest_overlaps = -sys.maxsize, sys.maxsize

    if recognized_rectangles:

        clusters = group_by_rectangle_area(list(recognized_rectangles), epsilon)

        for cluster_id, cluster in clusters.items():
            if cluster_id == -1:
                continue

            overlaps_num = rectangles_overlap(cluster)
            squares_num = compute_fitness(len(cluster), overlaps_num)
            if squares_num > best_fitness and smallest_overlaps > overlaps_num:
                best_fitness = squares_num
                smallest_overlaps = overlaps_num
                best_cluster = cluster

            if best_fitness == BOARD_SQUARES:
                break

        # best_cluster = max(clusters.values(), key=lambda item: len(item))
        # overlaps_num = rectangles_overlap(best_cluster)
        # best_fitness = compute_fitness(len(best_cluster), overlaps_num)

    if debug and best_cluster:
        draw_segmented(
            img,
            segmented,
            sorted_intersections,
            best_cluster,
            img_width,
            img_height,
        )
        return best_fitness

    return best_fitness


def do_recognize(img):
    def inner(inp, _):
        return scan(img, inp, debug=False)

    return inner


def func_generation(ga_instance):
    print(ga_instance.best_solution())


def recognize_board(file_path):
    img = cv.imread(file_path)

    # draw border which helps in case of board occupies the whole size of the picture for algorithm to be
    # able to detect board edges
    img = cv.copyMakeBorder(
        img, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=[255, 255, 255]
    )

    img_height, img_width, _ = img.shape

    # resized_img = resize_image(img, 400, 400)

    find_board = do_recognize(img)

    ga_instance = pygad.GA(
        num_generations=20,
        num_parents_mating=10,
        mutation_type="adaptive",
        fitness_func=find_board,
        mutation_probability=[0.5, 0.2],
        mutation_num_genes=[5, 2],
        mutation_percent_genes=[40, 20],
        crossover_probability=0.1,
        on_generation=func_generation,
        gene_space=[
            range(10, 100),  # threshold1 for Canny
            range(10, 200),  # threshold2 for Canny
            range(80, max(img_width, img_height) + 2),  # minimum points laying on line
            range(3, 11, 2),  # gauss
            range(3, 11, 2),  # gauss
            range(0, 5),  # maximum dispersion for a point of rectangle
            range(10, 80),  # minimum side length
            range(10, 300),  # maximum side length
            range(50, 105),  # epsilon for DBSCAN algorithm
            range(18, 31),
        ],
        gene_type=int,
        sol_per_pop=300,
        num_genes=10,
        stop_criteria="reach_64",
    )

    ga_instance.run()
    solution, _, _ = ga_instance.best_solution()

    scan(img, solution, debug=True)

    # ga_instance.plot_fitness()

    cv.imshow("Ready", img)
    cv.waitKey(0)
