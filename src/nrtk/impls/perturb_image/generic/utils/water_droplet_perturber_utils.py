"Utility functions for the Water Droplet Perturber"

from __future__ import annotations

import numpy as np

try:
    from scipy.special import binom

    scipy_available = True
except ImportError:
    scipy_available = False


class Bezier:
    """
    Class that computes the Bezier curve based on the segment information.
    Each curve is made of a series of segments that are initialized by the
    input points, angles, target radius and number of points needed for the
    Bezier interpolation.
    """

    def __init__(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        angle1: float,
        angle2: float,
        r: float = 0.3,
        num_points: int = 100,
    ) -> None:
        """Define segment parameters - points, angles, radius."""
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.r = r
        self.num_points = num_points
        self._calc_intermediate_points()

    def _calc_intermediate_points(self) -> None:
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        p = np.zeros((4, 2))
        p[0, :] = self.p1[:]
        p[3, :] = self.p2[:]
        p[1, :] = self.p1 + np.array([self.r * d * np.cos(self.angle1), self.r * d * np.sin(self.angle1)])
        p[2, :] = self.p2 + np.array(
            [self.r * d * np.cos(self.angle2 + np.pi), self.r * d * np.sin(self.angle2 + np.pi)],
        )
        self.p = p

    def get_curve(self) -> np.ndarray:
        """Returns curve information"""
        return self.bezier()

    def bezier(self) -> np.ndarray:
        """
        Draw Bézier curve by interpolating segments based on the
        Bernstein basis poynomial function.

        The Bezier curve equation is derived by combining the Bernstein basis polynomials with the control points:
            B(t) =  ∑(i=0 to n) B(i, n)(t) * P(i)
        """

        def _bernstein(n: int, k: int, t: np.ndarray) -> np.ndarray:
            """
            Bernstein basis polynomial function which is defined as:

            B(k, n)(t) = (n! / (k! * (n-k)!)) * t^k * (1 - t)^(n - k)

            n is the degree of the curve
            k is the index of the control point
            t is a parameter that varies from 0 to 1, defining the position along the curve.

            For n=2, the Bernstein polynomials are:
            - B(0, 2)(t) = (1 - t)^2
            - B(1, 2)(t) = 2t(1 - t)
            - B(2, 2)(t) = t^2

            For n=3, the Bernstein polynomials are:
            - B(0, 3)(t) = (1 - t)^3
            - B(1, 3)(t) = 3t(1 - t)^2
            - B(2, 3)(t) = 3t^2(1 - t)
            - B(3, 3)(t) = t^3.

            """
            return binom(n, k) * t**k * (1.0 - t) ** (n - k)  # pyright: ignore [reportPossiblyUnboundVariable]

        n = len(self.p)
        t = np.linspace(0, 1, num=self.num_points)
        curve = np.zeros((self.num_points, 2))
        for i in range(n):
            curve += np.outer(
                _bernstein(n=n - 1, k=i, t=t),
                self.p[i],
            )
        return curve


def ccw_sort(points: np.ndarray) -> np.ndarray:
    """Sorts points in counterclockwise order around a center point."""

    # Subtract original point from center point (position obtained
    # by calculating the mean)
    d = points - np.mean(points, axis=0)
    # Use atan2 to determine the angle taking into account the correct quadrant
    s = np.arctan2(d[:, 0], d[:, 1])
    # Return the sorted array of points.
    return points[np.argsort(s), :]


def get_bezier_curve(points: np.ndarray, rad: float = 0.2, edgy: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an array of *points*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest.
    """
    p = np.arctan(edgy) / np.pi + 0.5
    points = ccw_sort(points)
    points = np.append(points, np.atleast_2d(points[0, :]), axis=0)
    d = np.diff(points, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])

    def _threshold_angle(ang: np.ndarray) -> np.ndarray:
        return (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)

    ang = _threshold_angle(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    points = np.append(points, np.atleast_2d(ang).T, axis=1)

    def _get_curve(points: np.ndarray, r: float) -> np.ndarray:
        """Get the segments and curve data."""
        segments = list()
        for i in range(len(points) - 1):
            seg = Bezier(
                points[i, :2],
                points[i + 1, :2],
                points[i, 2],
                points[i + 1, 2],
                r,
            )
            segments.append(seg.get_curve())

        return np.concatenate(segments)

    c = _get_curve(points=points, r=rad)
    x, y = c.T
    return x, y


def get_random_points_within_min_dist(
    rng: np.random.Generator,
    n: int = 5,
    scale: float = 0.8,
    min_dst: float | None = None,
    recursive: int = 0,
) -> np.ndarray:
    """
    Recursive function to create *n* random points in the unit square,
    which are *min_dst* apart, then scale them.
    """
    min_dst = min_dst or 0.7 / n
    points = rng.random((n, 2))
    d = np.sqrt(np.sum(np.diff(ccw_sort(points), axis=0), axis=1) ** 2)
    if np.all(d >= min_dst) or recursive >= 200:
        return points * scale
    return get_random_points_within_min_dist(rng, n=n, scale=scale, min_dst=min_dst, recursive=recursive + 1)
