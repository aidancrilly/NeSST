import equinox as eqx
import interpax
import jax
import numpy as np
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

"""

Some wrappers to interpax to replace scipy's interp1d and RegularGridInterpolator

"""

_INTERPAX_METHOD = {
    "linear": "linear",
    "nearest": "nearest",
    "cubic": "cubic2",
    "quadratic": "cubic2",
    "slinear": "linear",
}


def safe_array(x):
    """Converts floats or lists to appropriate sized arrays"""
    if not hasattr(x, "ndim"):
        x = np.array(x)
    return np.atleast_1d(x)


def _unique_increasing(x_points, values, axis):
    """ENDF interpreted format can contain repeated abscissae"""
    if np.all(x_points[1:] > x_points[:-1]):
        return x_points, values
    x_points, unique_indices = np.unique(x_points, return_index=True)
    return x_points, np.take(values, unique_indices, axis=axis)


class Interpolator1D(eqx.Module):
    """interp1d-like callable which preserves the shape of its argument"""

    x: Array
    y: Array
    method: str = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    bounds_error: bool = eqx.field(static=True)
    fill_value: float = eqx.field(static=True)
    degenerate: bool = eqx.field(static=True)

    def __init__(self, x_points, values, method="linear", bounds_error=True, fill_value=jnp.nan, axis=-1):
        x_points = safe_array(x_points)
        assert x_points.ndim == 1
        values = np.asarray(values)
        self.axis = axis if axis is not None else -1
        self.degenerate = bool(np.all(np.isclose(x_points, x_points[0])))
        if not self.degenerate:
            x_points, values = _unique_increasing(x_points, values, self.axis)
        self.x = jnp.asarray(x_points)
        self.y = jnp.asarray(values)
        self.method = _INTERPAX_METHOD.get(method, method)
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value

    def __call__(self, x):
        x = jnp.asarray(x)
        xshape = x.shape
        xf = jnp.atleast_1d(x).reshape(-1)
        if self.degenerate:
            y = jnp.where(jnp.isclose(xf, self.x[0]), self.y[0], self.fill_value)
            return y.reshape(xshape)
        yv = jnp.moveaxis(self.y, self.axis, 0)
        y = interpax.interp1d(xf, self.x, yv, method=self.method, extrap=False)
        oob = (xf < self.x[0]) | (xf > self.x[-1])
        oob = oob.reshape((-1,) + (1,) * (y.ndim - 1))
        y = jnp.where(oob, jnp.nan if self.bounds_error else self.fill_value, jnp.nan_to_num(y, nan=self.fill_value))
        y = y.reshape(xshape + y.shape[1:])
        if self.y.ndim > 1 and len(xshape) > 0:
            ax = self.axis % self.y.ndim
            y = jnp.moveaxis(y, tuple(range(len(xshape))), tuple(range(ax, ax + len(xshape))))
        return y


class Interpolator2D(eqx.Module):
    """Regular grid bilinear interpolator over the outer product of its two arguments"""

    x: Array
    y: Array
    values: Array
    method: str = eqx.field(static=True)
    bounds_error: bool = eqx.field(static=True)
    fill_value: float = eqx.field(static=True)

    def __init__(self, x_points, y_points, values, method="linear", bounds_error=True, fill_value=jnp.nan):
        x_points = safe_array(x_points)
        y_points = safe_array(y_points)
        assert x_points.ndim == 1
        assert y_points.ndim == 1
        self.x = jnp.asarray(x_points)
        self.y = jnp.asarray(y_points)
        self.values = jnp.asarray(values)
        self.method = _INTERPAX_METHOD.get(method, method)
        self.bounds_error = bool(bounds_error)
        self.fill_value = fill_value

    def __call__(self, x, y):
        x = jnp.atleast_1d(jnp.asarray(x))
        y = jnp.atleast_1d(jnp.asarray(y))
        assert x.ndim == 1
        assert y.ndim == 1
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        f = interpax.interp2d(
            xx.reshape(-1), yy.reshape(-1), self.x, self.y, self.values, method=self.method, extrap=False
        )
        f = f.reshape(xx.shape)
        if self.bounds_error:
            oob = (xx < self.x[0]) | (xx > self.x[-1]) | (yy < self.y[0]) | (yy > self.y[-1])
            f = jnp.where(oob, jnp.nan, f)
        else:
            f = jnp.nan_to_num(f, nan=self.fill_value)
        return jnp.squeeze(f)


def interpolate_1d(x_points, values, method="linear", bounds_error=True, fill_value=jnp.nan, axis=None):
    return Interpolator1D(x_points, values, method=method, bounds_error=bounds_error, fill_value=fill_value, axis=axis)


def interpolate_2d(x_points, y_points, values, method="linear", bounds_error=True, fill_value=jnp.nan):
    return Interpolator2D(x_points, y_points, values, method=method, bounds_error=bounds_error, fill_value=fill_value)


def cumulative_trapezoid(y, x, initial=None):
    """jnp replacement for scipy.integrate.cumulative_trapezoid"""
    increments = 0.5 * (y[1:] + y[:-1]) * jnp.diff(x)
    cumulative = jnp.cumsum(increments)
    if initial is None:
        return cumulative
    return jnp.concatenate([jnp.full((1,), initial, dtype=cumulative.dtype), cumulative])


def uniform_filter1d(x, size):
    """jnp replacement for scipy.ndimage.uniform_filter1d with mode='constant', cval=0"""
    n = x.shape[0]
    lo = size // 2
    padded = jnp.concatenate([jnp.zeros(size), x, jnp.zeros(size)])
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(padded)])
    start = jnp.arange(n) + size - lo
    return (cumulative[start + size] - cumulative[start]) / size


def dynamic_uniform_filter1d(x, size, max_size):
    """uniform_filter1d for a traced window size, bounded above by max_size"""
    n = x.shape[0]
    lo = size // 2
    padded = jnp.concatenate([jnp.zeros(max_size), x, jnp.zeros(max_size)])
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(padded)])
    start = jnp.arange(n) + max_size - lo
    return (cumulative[start + size] - cumulative[start]) / size


def roll_zero(arr, n):
    """Roll a 1-D array by ``n`` positions, filling vacated entries with zero.

    Unlike ``jnp.roll``, this does not wrap around.  Positive ``n`` shifts
    towards later times; negative ``n`` shifts towards earlier times.
    Shifts larger than the array length return an all-zero array.
    """
    idx = jnp.arange(arr.shape[0]) - n
    return jnp.where((idx >= 0) & (idx < arr.shape[0]), arr[jnp.clip(idx, 0, arr.shape[0] - 1)], 0.0)


def legval(x, c):
    """jnp replacement for numpy.polynomial.legendre.legval with tensor=False"""
    c = jnp.asarray(c)
    n = c.shape[0]
    if n == 1:
        return c[0] * jnp.ones_like(x)
    P_prev = jnp.ones_like(x)
    P_curr = x
    total = c[0] * P_prev + c[1] * P_curr
    for i in range(2, n):
        P_prev, P_curr = P_curr, ((2 * i - 1) * x * P_curr - (i - 1) * P_prev) / i
        total = total + c[i] * P_curr
    return total
