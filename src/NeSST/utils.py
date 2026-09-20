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


def Ecentres_to_edges(Ecentres):
    """Convert energy bin centres to edges.

    The outer half-widths are mirrored from the first and last interval, so a
    uniform grid of centres maps back to exactly the grid it came from.

    Args:
        Ecentres (array): energy bin centres in eV

    Returns:
        tuple: energy bin edges in eV, and the bin widths
    """
    Ecentres = jnp.asarray(Ecentres)
    inner = 0.5 * (Ecentres[:-1] + Ecentres[1:])
    first = Ecentres[0] - 0.5 * (Ecentres[1] - Ecentres[0])
    last = Ecentres[-1] + 0.5 * (Ecentres[-1] - Ecentres[-2])
    Eedges = jnp.concatenate([first[None], inner, last[None]])
    return Eedges, jnp.diff(Eedges)


def energy_bin_edges(Ecentres):
    """Ecentres_to_edges for an energy grid, with the mirrored outer edges kept
    non-negative.  A first bin whose mirrored edge falls below zero can only
    mean a bin starting at zero; letting it through gives negative energies.

    Args:
        Ecentres (array): energy bin centres in eV

    Returns:
        tuple: energy bin edges in eV, and the bin widths
    """
    Eedges, _ = Ecentres_to_edges(Ecentres)
    Eedges = jnp.clip(Eedges, 0.0, None)
    return Eedges, jnp.diff(Eedges)


def midpoint_subnodes(lo, hi, N):
    """Midpoint rule sub-division of a set of bins.

    Args:
        lo (array): lower bin edges, any shape
        hi (array): upper bin edges, same shape as lo
        N (int): number of sub-divisions per bin

    Returns:
        tuple: nodes with a trailing axis of length N, and the sub-node width
    """
    width = (hi - lo) / N
    offsets = jnp.arange(N) + 0.5
    return lo[..., None] + offsets * width[..., None], width


# Elements per tile when the outgoing energy axis is blocked.  The bin averaged
# kernels are memory bandwidth bound, so evaluating the whole outgoing grid at
# once is both the largest allocation and the slowest option once a bin carries
# sub-nodes.  Half a million elements keeps a tile near cache and was the best
# compromise measured across the elastic and ion velocity kernels.
TILE_ELEMENTS = 1 << 19


def integrate_sub_nodes(sub_node, zero, N):
    """Sum a sub-node integrand, scanning rather than materialising all N of them

    Args:
        sub_node (callable): the integrand at sub-node index k
        zero (array): accumulator of the shape sub_node returns
        N (int): number of sub-nodes

    Returns:
        array: the summed integrand
    """
    if N == 1:
        return sub_node(0)
    total, _ = jax.lax.scan(lambda acc, k: (acc + sub_node(k), None), zero, jnp.arange(N))
    return total


def map_outgoing_bins(outgoing_bin, Eout_edges, row_elements, N):
    """Apply outgoing_bin over the outgoing grid, blocking it when a tile is large

    Blocking only pays once a bin carries sub-nodes.  At N = 1 evaluating the
    whole grid at once is already minimal and chunking just adds overhead.

    Args:
        outgoing_bin (callable): maps one (lower, upper) edge pair to its row
        Eout_edges (array): outgoing bin edges
        row_elements (int): elements one row of the result works over
        N (int): sub-nodes per bin

    Returns:
        array: the rows stacked along the outgoing axis
    """
    edges = (Eout_edges[:-1], Eout_edges[1:])
    n_out = Eout_edges.shape[0] - 1
    batch = max(1, TILE_ELEMENTS // max(row_elements, 1))
    if N == 1 or batch >= n_out:
        return jax.vmap(outgoing_bin)(edges)
    return jax.lax.map(outgoing_bin, edges, batch_size=batch)
