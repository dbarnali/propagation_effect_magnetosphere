import distutils.core
import Cython.Build
import numpy
from setuptools import setup, Extension

include_gsl_dir = "/usr/include/"
lib_gsl_dir = "/usr/lib/x86_64-linux-gnu/"

#########################################3
filename1	='propagation_effect_tools_initial_condition_3D_complex_IM_cython'
filename2	='propagation_effect_tools_solve_integration_3D_complex_IM_cython'
filename3	='propagation_effect_tools_density_func'

ext_modules = [Extension(filename1,sources=[filename1+'.pyx'],include_dirs=[numpy.get_include(), include_gsl_dir],libraries=["m",'gsl',"gslcblas"],library_dirs=[lib_gsl_dir]),Extension(filename2,sources=[filename2+'.pyx'],include_dirs=[numpy.get_include(), include_gsl_dir]),Extension(filename3,sources=[filename3+'.pyx'],include_dirs=[numpy.get_include(), include_gsl_dir],libraries=["m",'gsl',"gslcblas"],library_dirs=[lib_gsl_dir])]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3

setup(
    setup_requires=['setuptools>18.0','cython'],
    ext_modules = ext_modules)
#python3 setup.py build_ext --inplace

