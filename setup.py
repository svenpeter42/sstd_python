#!/usr/bin/env python
# base on https://github.com/cython/cython/wiki/PackageHierarchy
import sys, os
from distutils.core import setup
from distutils.extension import Extension


# we'd better have Cython installed, or it's a no-go
try:
	from Cython.Distutils import build_ext
except:
	print("You don't seem to have Cython installed. Please get a")
	print("copy from www.cython.org and install it")
	sys.exit(1)

def scandir(dir_, files=[]):
	for f in os.listdir(dir_):
		path = os.path.join(dir_, f)
		if os.path.isfile(path) and path.endswith(".pyx"):
			files.append(path.replace(os.path.sep, ".")[:-4])
		elif os.path.isdir(path):
			scandir(path, files)
	return files

def make_ext(extname):
	extpath = extname.replace(".", os.path.sep) + ".pyx"
	return Extension(
		extname,
		[extpath],
		include_dirs = ["."],
		extra_compile_args = ["-O3", "-Wall"],
		extra_link_args = ['-g'],
		)

setup(
	name="SSTD",
	packages=["SSTD"],
    ext_modules = [make_ext(n) for n in scandir("SSTD")],
	cmdclass = {'build_ext': build_ext}
)

