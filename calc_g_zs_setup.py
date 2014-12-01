#!/usr/bin/env python
import sys
import shutil
import os

if len(sys.argv) == 1:
    sys.argv.append('build')

#from distutils.core import setup, Extension
from setuptools import setup, Extension
from distutils.command.build_ext import build_ext


from distutils.util import get_platform
platform = '.%s-%s'%(get_platform(),sys.version[:3])

extra_compile_args =  {'msvc': ['/EHsc']}
extra_link_args =  {}

class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in extra_compile_args:
           for e in self.extensions:
               e.extra_compile_args = extra_compile_args[c]
        if c in extra_link_args:
            for e in self.extensions:
                e.extra_link_args = extra_link_args[c]
        build_ext.build_extensions(self)
        
from numpy import get_include
calc_g_zs_module = Extension('calc_g_zs_cex',
                             sources=['calc_g_zs_cex.c'],
                             include_dirs=[get_include()],
                             )
                     
dist = setup(
        name = 'SCF1d',
        version = 0.01,
        author='Richard Sheridan',
        author_email='richard.sheridan@nist.gov',
        description='1-D end tethered polymer modeling',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: Public Domain',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Physics',
            ],
        ext_modules = [calc_g_zs_module],
        install_requires = ['numpy>=1.8.0','scipy>=0.11.0'],
        cmdclass = {'build_ext': build_ext_subclass },
        )

shutil.copy('build\lib{platform}\calc_g_zs_cex.pyd'.format(platform=platform),
            'calc_g_zs_cex.pyd')