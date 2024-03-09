from scipy.interpolate import RegularGridInterpolator
from numpy import nan

"""

Some wrappers to RegularGridInterpolator to replace interp1d and interp2d

"""

def interpolate_1d(x_points,values,method='linear',bounds_error=True,fill_value=nan):
    if(x_points.shape == values.shape):
        return RegularGridInterpolator((x_points),values,method,bounds_error,fill_value)
    else:
        return TypeError

def interpolate_2d(x_points,y_points,values,method='linear',bounds_error=True,fill_value=nan):
    if((x_points.shape[0] == values.shape[0]) & (y_points.shape[0] == values.shape[1])):
        return RegularGridInterpolator((x_points,y_points),values,method,bounds_error,fill_value)
    else:
        return TypeError
