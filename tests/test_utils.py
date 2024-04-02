import pytest
import numpy as np
import NeSST as nst

def test_linear_1D_interpolate():
    x = np.array([0.0,1.0])
    y = np.array([0.0,1.0])
    f = nst.utils.interpolate_1d(x,y)
    x_test = np.random.rand(100)
    y_test = f(x_test)
    assert np.all(np.isclose(x_test,y_test))

def test_linear_1D_interpolate_nonunique():
    x = np.array([0.0,0.0,1.0,1.0])
    y = np.array([0.0,0.0,1.0,1.0])
    f = nst.utils.interpolate_1d(x,y)
    x_test = np.random.rand(100)
    y_test = f(x_test)
    assert np.all(np.isclose(x_test,y_test))
    
def test_linear_1D_interpolate_dimension_matching():
    x = np.array([0.0,1.0])
    y = np.array([0.0,1.0])
    f = nst.utils.interpolate_1d(x,y)
    x_test = np.random.rand(100).reshape(50,2)
    y_test = f(x_test)
    assert np.all(np.isclose(x_test,y_test))
    assert x_test.shape == y_test.shape

def test_linear_2D_interpolate():
    def z_func(x,y):
        return x-0.5*y
    x = np.array([0.0,1.0])
    y = np.array([0.0,1.0])
    xx,yy = np.meshgrid(x,y, indexing='ij')
    z = z_func(xx,yy)
    f = nst.utils.interpolate_2d(x,y,z)
    # Equal sizes
    x_test = np.random.rand(50)
    y_test = np.random.rand(50)
    z_test = f(x_test,y_test)
    xx,yy = np.meshgrid(x_test,y_test, indexing='ij')
    zz = z_func(xx,yy)
    assert zz.shape == z_test.shape
    assert np.all(np.isclose(zz,z_test))
    # Unequal sizes
    x_test = np.random.rand(50)
    y_test = np.random.rand(3)
    z_test = f(x_test,y_test)
    xx,yy = np.meshgrid(x_test,y_test, indexing='ij')
    zz = z_func(xx,yy)
    assert np.all(np.isclose(zz,z_test))
    assert zz.shape == z_test.shape