from scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np

"""

Some wrappers to RegularGridInterpolator to replace interp1d and interp2d

"""

def safe_array(x):
    """Converts floats or lists to appropriate sized arrays"""
    if not hasattr(x,'ndim'):
        x = np.array(x)
    return np.atleast_1d(x)

def interpolate_1d(x_points,values,method='linear',bounds_error=True,fill_value=np.nan,axis=None):
    x_points = safe_array(x_points)
    assert x_points.ndim == 1
    if(axis is None):
        axis = -1
    return interp1d(x_points,values,kind=method,bounds_error=bounds_error,fill_value=fill_value,axis=axis)
    """
    Below is some future proofing where we attempt to reclaim some of interp1d's behaviour with RegularGridInterpolator

    It is less performant so is not used, but if interp1d becomes deprecated it might be needed...
    """
    if(values.ndim == 1):
        if(np.all(np.isclose(x_points, x_points[0]))):
            # Sometimes the array is just a series of the same points...
            def point_interpolant(x):
                y = fill_value*np.ones_like(x)
                y[np.isclose(x, x_points[0])] = values[0]
                return y
            return point_interpolant
        else:
            # Sometimes ENDF interpreted format has repeated elements...
            if(not np.all(x_points[1:] > x_points[:-1])):
                x_points,unique_indices = np.unique(x_points,return_index=True)
                values = values[unique_indices]
            # We often want to call 1D interpolators with non 1D shaped inputs
            RGI = RegularGridInterpolator((x_points,),values,method,bounds_error,fill_value)
            def dimension_matching_interp(x):
                x = np.atleast_1d(x)
                xshape = x.shape
                y = RGI(x.flatten())
                return y.reshape(xshape)
            return dimension_matching_interp
    else:
        # Currently doesn't seem to be axis argument support for RegularGridInterpolator...
        return interp1d(x_points,values,kind=method,bounds_error=bounds_error,fill_value=fill_value,axis=axis)

def interpolate_2d(x_points,y_points,values,method='linear',bounds_error=True,fill_value=np.nan):
    x_points = safe_array(x_points)
    y_points = safe_array(y_points)
    assert x_points.ndim == 1
    assert y_points.ndim == 1
    RGI = RegularGridInterpolator((x_points,y_points),values,method,bounds_error,fill_value)
    def dimension_matching_interp(x,y):
        x = safe_array(x)
        y = safe_array(y)
        assert x.ndim == 1
        assert y.ndim == 1
        xx,yy = np.meshgrid(x,y,indexing='ij')
        f = RGI((xx,yy))
        return np.squeeze(f)
    return dimension_matching_interp
