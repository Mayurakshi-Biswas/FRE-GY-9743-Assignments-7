import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving piecewise-quadratic interpolator for pointwise IFRs.

    Construction (after Schumaker, 1983; design criteria from Hagan & West, 2006):
      * Secants s_i = (f_{i+1} - f_i) / h_i on each node interval.
      * A slope is chosen at every node: a weighted three-point estimate at
        interior nodes and a one-sided three-point estimate at the two ends,
        clipped so that each interval satisfies
            left slope <= s_i <= right slope        (convexity)
            slopes share the sign of s_i            (monotonicity).
        If no single slope at an interior node satisfies both neighbouring
        intervals, each side uses its own secant (a derivative jump there).
      * On interval i with end slopes a, b:
            a < s_i < b  -> two quadratic pieces joined at the point where the
                            two end tangents meet; f' runs a -> s_i -> b
                            (piecewise linear and non-decreasing).
            otherwise    -> straight line with slope s_i.
    Guarantee: continuous, passes through every node, monotone and convex on
    each node interval. f' may jump at nodes and at the joins with FLAT
    extrapolation. integrate() is exact (closed form).
    """

    _TOL = 1e-12  # relative tolerance for the strict test a < s < b

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

        self.x_ = np.asarray(self.axis1_, dtype=float)
        self.f_ = np.asarray(self.values_, dtype=float)
        n = self.length_
        if n >= 2 and np.any(np.diff(self.x_) <= 0.0):
            raise ValueError("MONOTONE_CONVEX requires strictly increasing node times.")

        # pieces_[i] = None for a straight line, else (a, b, xi) for two quadratics.
        self.pieces_ = []
        self.cum_area_ = np.zeros(n)  # integral of f from x_0 to x_i
        if n == 1:
            return

        h = np.diff(self.x_)
        s = np.diff(self.f_) / h
        self.h_, self.s_ = h, s

        d_left, d_right = self._node_slopes(h, s)  # slope leaving / arriving at each node

        for i in range(n - 1):
            a, b, si = d_left[i], d_right[i + 1], s[i]
            tol = self._TOL * max(1.0, abs(a), abs(b), abs(si))
            if a < si - tol and b > si + tol:
                lam = (b - si) / (b - a)          # strictly inside (0, 1)
                xi = self.x_[i] + lam * h[i]      # where the two end tangents meet
                self.pieces_.append((a, b, xi))
            else:
                self.pieces_.append(None)         # straight line with slope s_i
            self.cum_area_[i + 1] = self.cum_area_[i] + self._area_in_interval(i, self.x_[i + 1])

    @staticmethod
    def _node_slopes(h: np.ndarray, s: np.ndarray):
        """Return (d_left, d_right): slope leaving node i to the right and arriving at node i from the left."""
        n = len(s) + 1
        d_left = np.zeros(n)
        d_right = np.zeros(n)
        if n == 2:  # one interval: straight line
            d_left[0] = d_right[1] = s[0]
            return d_left, d_right

        # First node: one-sided three-point estimate; must satisfy d <= s_0, and d >= 0 if s_0 >= 0.
        d0 = s[0] - h[0] * (s[1] - s[0]) / (h[0] + h[1])
        d0 = min(d0, s[0])
        if s[0] >= 0.0:
            d0 = max(d0, 0.0)
        d_left[0] = d0

        # Last node: one-sided three-point estimate; must satisfy d >= s_last, and d <= 0 if s_last <= 0.
        dn = s[-1] + h[-1] * (s[-1] - s[-2]) / (h[-2] + h[-1])
        dn = max(dn, s[-1])
        if s[-1] <= 0.0:
            dn = min(dn, 0.0)
        d_right[-1] = dn

        # Interior nodes: weighted three-point estimate, clipped to the range both neighbours accept.
        for i in range(1, n - 1):
            est = (h[i] * s[i - 1] + h[i - 1] * s[i]) / (h[i - 1] + h[i])
            lo = max(s[i - 1], 0.0) if s[i] >= 0.0 else s[i - 1]
            hi = min(s[i], 0.0) if s[i - 1] <= 0.0 else s[i]
            if lo <= hi:
                d_left[i] = d_right[i] = min(max(est, lo), hi)
            else:  # no shared slope works: each side uses its own secant
                d_right[i] = s[i - 1]
                d_left[i] = s[i]
        return d_left, d_right

    def _value_in_interval(self, i: int, x: float) -> float:
        x0, f0, si = self.x_[i], self.f_[i], self.s_[i]
        piece = self.pieces_[i]
        if piece is None:
            return f0 + si * (x - x0)
        a, b, xi = piece
        if x <= xi:
            u = x - x0
            return f0 + a * u + (si - a) / (2.0 * (xi - x0)) * u * u
        f_xi = f0 + 0.5 * (a + si) * (xi - x0)
        w = x - xi
        return f_xi + si * w + (b - si) / (2.0 * (self.x_[i + 1] - xi)) * w * w

    def _area_in_interval(self, i: int, x: float) -> float:
        """Exact integral of f from x_i to x, for x in [x_i, x_{i+1}]."""
        x0, f0, si = self.x_[i], self.f_[i], self.s_[i]
        piece = self.pieces_[i]
        if piece is None:
            u = x - x0
            return f0 * u + 0.5 * si * u * u
        a, b, xi = piece
        w1 = xi - x0
        if x <= xi:
            u = x - x0
            return f0 * u + 0.5 * a * u * u + (si - a) * u ** 3 / (6.0 * w1)
        area1 = f0 * w1 + 0.5 * a * w1 * w1 + (si - a) * w1 * w1 / 6.0  # full first piece
        f_xi = f0 + 0.5 * (a + si) * w1
        w = x - xi
        return area1 + f_xi * w + 0.5 * si * w * w + (b - si) * w ** 3 / (6.0 * (self.x_[i + 1] - xi))

    def _locate(self, x: float) -> int:
        return min(int(np.searchsorted(self.x_, x, side='right')) - 1, self.length_ - 2)

    def _antiderivative(self, x: float) -> float:
        """F(x) = integral of f from x_0 to x, with FLAT extrapolation on both sides."""
        x0 = self.x_[0]
        if self.length_ == 1 or x <= x0:
            return self.f_[0] * (x - x0)
        xn = self.x_[-1]
        if x >= xn:
            return self.cum_area_[-1] + self.f_[-1] * (x - xn)
        i = self._locate(x)
        return self.cum_area_[i] + self._area_in_interval(i, x)

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.x_[0]:
            return float(self.f_[0])
        if x >= self.x_[-1]:
            return float(self.f_[-1])
        return float(self._value_in_interval(self._locate(x), x))

    def integrate(self, start_x: float, end_x: float) -> float:
        return float(self._antiderivative(end_x) - self._antiderivative(start_x))

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')