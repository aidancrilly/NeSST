from scipy.interpolate import RegularGridInterpolator,interp1d
import numpy as np

"""

Some wrappers to RegularGridInterpolator to replace interp1d and interp2d

"""

def interpolate_1d(x_points,values,method='linear',bounds_error=True,fill_value=np.nan,axis=None):
    assert x_points.ndim == 1
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
    assert x_points.ndim == 1
    assert y_points.ndim == 1
    RGI = RegularGridInterpolator((x_points,y_points),values,method,bounds_error,fill_value)
    def dimension_matching_interp(x,y):
        assert x.ndim == 1
        assert y.ndim == 1
        xx,yy = np.meshgrid(x,y,indexing='ij')
        f = RGI((xx,yy))
        return f
    return dimension_matching_interp
