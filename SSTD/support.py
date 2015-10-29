# Copyright 2014, 2015 Sven Peter <sven.peter@iwr.uni-heidelberg.de>
# Licensed under the terms of the GNU GPL, version 2
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

import numpy as np
from warnings import warn

def toepliz_conv_2d_sparse(shape_in, shape_out, x, y, val = 1.0):
	m, n = shape_out

	mtx = np.zeros((m*n, m*n))

	delta_x = shape_in[0] - shape_out[0]
	delta_y = shape_in[1] - shape_out[1]


	row = 0
	for i_m in xrange(m):
		for i_n in xrange(n):
			col_x = (((delta_x + i_m - x) + shape_out[0]/2) % shape_out[0])
			col_y = (((delta_y + i_n - y) + shape_out[1]/2) % shape_out[1])

			col = col_x * n + col_y

			mtx[row, col] = val
			row += 1

	return mtx


def sanitize_array(a, dtype, ndim, force_contig):
	if len(a.shape) != ndim:
		warn("sanitize_array: a.shape = %s != %d  = ndim" % (a.shape, ndim))

		if ndim == 1: a = a.reshape(-1)
		elif ndim == 2: a = np.atleast_2d(a)
		elif ndim == 3: a = np.atleast_3d(a)
		else: raise ValueError("sanitize_array: ndim = %d not support" % ndim)

	if force_contig:
		if not a.flags['C_CONTIGUOUS']:
			warn("sanitize_array: input array not already c contiguous")
			a = np.ascontiguousarray(a)

	if a.dtype != dtype:
		warn("sanitize_array: input array is not %s but %s" % (a.dtype, dtype))
		a = a.astype(dtype)

	return a
