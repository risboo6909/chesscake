from functools import lru_cache


class PointsInArea(object):
    def __init__(self, points):
        self.sorted_x = sorted([point for point in points], key=lambda p: p[0][0])
        self.sorted_y = sorted([point for point in points], key=lambda p: p[0][1])

    @lru_cache
    def get_points(self, x, y, r):

        points_x = set()
        points_y = set()

        for p in self.sorted_x:
            if p[0][0] < x - r:
                continue
            if p[0][0] > x + r:
                break
            points_x.add(tuple(p[0]))

        for p in self.sorted_y:
            if p[0][1] < y - r:
                continue
            if p[0][1] > y + r:
                break
            points_y.add(tuple(p[0]))

        return points_x.intersection(points_y)

    def has_points(self, x, y, r):
        return len(self.get_points(x, y, r)) > 0
