from operator import itemgetter
from functools import lru_cache, reduce


class PointsInArea(object):
    def __init__(self, points):
        # points = [(x, y), (x, y), ...]
        self.sorted_x = sorted([point for point in points], key=itemgetter(0))
        self.sorted_y = sorted([point for point in points], key=itemgetter(1))

    @lru_cache
    def get_points(self, x, y, r):

        points_x = set()
        points_y = set()

        for p in self.sorted_x:
            if p[0] < x - r:
                continue
            if p[0] > x + r:
                break
            points_x.add(p)

        for p in self.sorted_y:
            if p[1] < y - r:
                continue
            if p[1] > y + r:
                break
            points_y.add(p)

        return points_x.intersection(points_y)

    def has_points(self, x, y, r):
        return len(self.get_points(x, y, r)) > 0

    def get_center_of_mass(self, x, y, r):
        points_in_area = self.get_points(x, y, r)
        sum_x, sum_y = reduce(
            lambda acc, item: (acc[0] + item[0], acc[1] + item[1]),
            points_in_area,
            (0, 0),
        )

        l = len(points_in_area)
        return sum_x / l, sum_y / l, points_in_area
