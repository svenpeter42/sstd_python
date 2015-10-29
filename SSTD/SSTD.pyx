# Copyright 2014, 2015 Sven Peter <sven.peter@iwr.uni-heidelberg.de>
# Licensed under the terms of the GNU GPL, version 2
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt

from __future__ import print_function

import numpy as np
cimport numpy as np

import scipy.signal as sig
import scipy.optimize as opt
import scipy.sparse as spmtx
import skimage.restoration

from cpython cimport bool
from cython.view cimport array as cvarray


import sys

from support import toepliz_conv_2d_sparse, sanitize_array



SSTD_FLOAT_np = np.float64
ctypedef np.float64_t SSTD_FLOAT_t

SSTD_UINT_np = np.int
ctypedef np.int_t SSTD_UINT_t

SSTD_INT_np = np.int
ctypedef np.int_t SSTD_INT_t

cdef SSTD_FLOAT_t[:,:] estimate_bg(SSTD_FLOAT_t[:,:] X, unsigned int n_rows, unsigned int n_cols, unsigned int n_iter = 100, tv_weight = 1):
	cdef unsigned int i = 0

	# get data = X = AY with rank(A) = 1, rank(Y) = 1

	# init Y to rand, solve for A
	cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Y = sanitize_array(np.random.randn(X.shape[1]).reshape(1, -1), SSTD_FLOAT_np, 2, True)
	cdef np.ndarray[SSTD_FLOAT_t, ndim=2] A = sanitize_array(np.linalg.lstsq(Y.T, X.T)[0], SSTD_FLOAT_np, 2, True)

	for i in range(n_iter):
		# multiplicative update rules + imm TV denoising of A
		A = np.divide(np.dot(X, Y.T), np.dot(Y, Y.T))
		norm = np.linalg.norm(A)
		A = skimage.restoration.denoise_tv_chambolle((1/(1.0 * norm)) * A.reshape((n_rows, n_cols)), weight = 10).reshape((-1,1))
		Y = np.divide(np.dot(A.T, X), np.dot(A.T, A))

	cdef np.ndarray[SSTD_FLOAT_t, ndim=2] res = np.dot(A, Y)
	cdef np.ndarray[SSTD_FLOAT_t, ndim=2] res_divide = res.copy()
	res_divide[np.where(res_divide == 0)] = 1

	return np.divide(X - res, res_divide)

cdef class SSTD(object):
	cdef unsigned int n_rows
	cdef unsigned int n_cols
	cdef unsigned int n_frames

	cdef unsigned int n_basis
	cdef unsigned int n_filter_space
	cdef unsigned int n_filter_time

	cdef unsigned int n_max_spikes
	cdef unsigned int learn_spikes_n_iter

	cdef bool do_bg
	cdef bool did_bg

	cdef SSTD_FLOAT_t[:,:] input_vanilla
	cdef SSTD_FLOAT_t[:,:] input_nobg

	cdef SSTD_FLOAT_t[:,:,:] filter_space
	cdef SSTD_FLOAT_t[:,:] filter_time

	# matrix D in the paper
	cdef SSTD_INT_t[:] D_flt
	cdef SSTD_UINT_t[:] D_x
	cdef SSTD_UINT_t[:] D_y
	cdef SSTD_FLOAT_t[:] D_val

	# matrix U in the paper
	cdef SSTD_INT_t[:] U_flt
	cdef SSTD_UINT_t[:] U_basis
	cdef SSTD_UINT_t[:] U_frame
	cdef SSTD_FLOAT_t[:] U_val

	cdef unsigned int current_iteration

	def __init__(self, input_, n_basis, n_filter_time, len_filter_time, n_filter_space, rows_filter_space, cols_filter_space, do_bg):
		self.do_bg = do_bg
		self.did_bg = False

		self.n_filter_space = n_filter_space
		self.n_filter_time = n_filter_time
		self.n_basis = n_basis

		input_ = sanitize_array(input_, SSTD_FLOAT_np, 3, True)

		self.n_rows, self.n_cols, self.n_frames = input_.shape

		self.filter_space = np.zeros((n_filter_space, rows_filter_space, cols_filter_space), dtype=SSTD_FLOAT_np, order='C')
		self.filter_time = np.zeros((n_filter_time, len_filter_time), dtype=SSTD_FLOAT_np, order='C')

		self.input_vanilla = input_.reshape(self.n_rows*self.n_cols, self.n_frames)
		self.current_iteration = 0

		if do_bg:
			self.input_nobg = np.zeros_like(self.input_vanilla)

		self.n_max_spikes = 10000
		self.learn_spikes_n_iter = 10000

		self.U_basis = np.zeros(self.n_max_spikes, dtype=SSTD_UINT_np)
		self.U_frame = np.zeros(self.n_max_spikes, dtype=SSTD_UINT_np)
		self.U_flt = np.zeros(self.n_max_spikes, dtype=SSTD_INT_np) - 1
		self.U_val = np.zeros(self.n_max_spikes, dtype=SSTD_FLOAT_np)

		self.D_flt = np.zeros(self.n_basis, dtype=SSTD_INT_np) - 1
		self.D_x = np.zeros(self.n_basis, dtype=SSTD_UINT_np)
		self.D_y = np.zeros(self.n_basis, dtype=SSTD_UINT_np)
		self.D_val = np.zeros(self.n_basis, dtype=SSTD_FLOAT_np)


		self.init_flt_space_all()
		self.init_time_flt_all()
		self.init_spikes()

	def init_flt_space(self, x, y):
		'''dim = [x, y]
		tmp = [sig.get_window(('gaussian', np.random.random() * d/3.), d) for d in dim]
		flt = np.array(np.mat(tmp[0]).T * np.mat(tmp[1]))'''
		flt = np.random.randn(x, y)
		flt = np.absolute(flt)
		flt = np.divide(flt, np.sum(flt))
		flt = np.divide(flt, np.linalg.norm(flt))
		return flt

	def init_flt_space_all(self):
		cdef unsigned int i
		cdef SSTD_FLOAT_t[:,:] flt
		for i in range(self.n_filter_space):
			flt = self.init_flt_space(self.filter_space.shape[1], self.filter_space.shape[2])
			self.filter_space[i, :, :] = flt

	def init_time_flt(self, len_):
		flt = np.random.randn(len_)
		flt[np.where(flt < 0)] = 0
		flt[0] = 2*np.max(flt)
		flt[::-1] = np.cumsum(flt[::-1])
		flt /= float(np.linalg.norm(flt))
		return flt

	def init_time_flt_all(self):
		cdef SSTD_FLOAT_t[:] flt
		for i in range(self.n_filter_time):
			flt = self.init_time_flt(self.filter_time.shape[1])
			self.filter_time[i, :] = flt

	def init_spikes(self):
		cdef unsigned int i_basis, i_filter, rnd, idx

		idx = 0
		for i_basis in range(self.n_basis):
			for i_filter in range(self.n_filter_time):
				if self.n_frames == 1:
					rnds = [0]
				else:	
					rnds = np.random.randint(0, self.n_frames, 5)

				for rnd in rnds:
					self.U_val[idx] = 1
					self.U_basis[idx] = i_basis
					self.U_flt[idx] = i_filter
					self.U_frame[idx] = rnd
					idx += 1

	def learn_filter_time(self):
		cdef np.ndarray[SSTD_FLOAT_t, ndim=3] Dconv = self._fc_Dconv().reshape((self.n_rows, self.n_cols, self.n_basis))
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Uconv = self._fc_Uconv()
		cdef SSTD_FLOAT_t[:,:,:] input_nobg_3d = np.asarray(self.input_nobg).reshape((self.n_rows, self.n_cols, self.n_frames))
		cdef SSTD_UINT_t[:] flt_start = np.zeros(self.n_basis, dtype=SSTD_UINT_np)
		cdef SSTD_FLOAT_t[:] flt
		cdef SSTD_FLOAT_t[:,:] flt_space

		cdef unsigned int flt_len = self.filter_time.shape[1]
		cdef unsigned int vectorized_flt_len = self.n_filter_time * flt_len

		cdef unsigned int col_offset
		cdef unsigned int row_offset
		cdef unsigned int frame, frame_real, x, y, x_real, y_real

		if self.n_frames == 1:
			self.filter_time[0, 0] = 1

		n_outputs = self.n_frames * self.n_rows * self.n_cols
		A = spmtx.dok_matrix((n_outputs, vectorized_flt_len))
		b = spmtx.dok_matrix((n_outputs, 1))

		print("    computing filter-output relations")
		for i_spike in range(self.n_max_spikes):
			if self.U_flt[i_spike] < 0: break

			i_basis = self.U_basis[i_spike]

			(x_start, y_start, flt_space) = self._fc_space(i_basis)
			x_len = flt_space.shape[0]
			y_len = flt_space.shape[1]

			for frame in range(flt_len):
				frame_real = self.U_frame[i_spike] + frame
				if frame_real >= self.n_frames: break
				for x in range(x_len):
					for y in range(y_len):
						x_real = x_start + x
						y_real = y_start + y

						col_offset = frame_real * self.n_cols * self.n_rows + x_real * self.n_cols + y_real
						row_offset = self.U_flt[i_spike] * flt_len + frame
						b[col_offset, 0] = input_nobg_3d[x_real, y_real, frame_real]
						A[col_offset, row_offset] += flt_space[x, y]


		print("    computing gram matrix")
		b = np.asarray(np.dot(A.T, b).todense()).reshape(-1)
		A = np.dot(A.T, A).todense()

		print("    finding nnls solution")
		x2 = opt.nnls(A, b)[0]

		flt = np.zeros(flt_len)
		for i_filter in range(self.n_filter_time):
			flt = x2[flt_start[i_filter]:flt_start[i_filter] + flt_len]
			norm = np.linalg.norm(flt)

			if norm == 0:
				flt = self.init_time_flt(flt_len)
			else:
				flt /= norm

			self.filter_time[i_filter, :] = flt



	def learn_filter_space(self):
		#cdef np.ndarray[SSTD_FLOAT_t, ndim=1] b
		#cdef np.ndarray[SSTD_FLOAT_t, ndim=1] x
		cdef np.ndarray[SSTD_FLOAT_t, ndim=3] Dconv = self._fc_Dconv().reshape((self.n_rows, self.n_cols, self.n_basis))
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Uconv = self._fc_Uconv()
		cdef np.ndarray[SSTD_UINT_t, ndim=2] flt_start = np.zeros((self.n_basis, 2), dtype=SSTD_UINT_np)
		cdef SSTD_FLOAT_t[:,:,:] input_nobg_3d = np.asarray(self.input_nobg).reshape((self.n_rows, self.n_cols, self.n_frames))
		cdef SSTD_FLOAT_t[:,:,:] slice_

		cdef unsigned int i_filter
		cdef unsigned int offset = 0
		cdef SSTD_FLOAT_t[:,:] flt

		cdef unsigned int flt_len = self.filter_space.shape[1] * self.filter_space.shape[2]
		cdef unsigned int vectorized_flt_len = self.n_filter_space * self.filter_space.shape[1] * self.filter_space.shape[2]

		cdef unsigned int col_offset
		cdef unsigned int row_offset
		cdef unsigned int frame, frame_real, x, y, x_real, y_real

		n_outputs = self.n_frames * self.n_rows * self.n_cols
		A = spmtx.dok_matrix((n_outputs, vectorized_flt_len))
		b = spmtx.dok_matrix((n_outputs, 1))

		print("    computing filter-output relations")
		print(("      %04d/%04d" % (0, self.n_basis)), end='')
		sys.stdout.flush()

		for i_basis in range(self.n_basis):
			print(("\r      %04d/%04d start" % (i_basis, self.n_basis)), end='')
			sys.stdout.flush()
			frames = np.where(Uconv[:, i_basis])[0]
			print(("\r      %04d/%04d start, Uconv found" % (i_basis, self.n_basis)), end='')
			sys.stdout.flush()

			i_filter = self.U_flt[i_basis]
			if i_filter < 0: continue

			(x_start, y_start, flt) = self._fc_space(i_basis)
			x_len = flt.shape[0]
			y_len = flt.shape[1]
			y_len_real = self.filter_space.shape[1]

			for frame in frames:
				for x in range(x_len):
					for y in range(y_len):
						x_real = x_start + x
						y_real = y_start + y

						row_offset = i_filter * flt_len + x * y_len_real + y
						col_offset = frame * self.n_cols * self.n_rows + x_real * self.n_cols + y_real

						b[col_offset, 0] = input_nobg_3d[x_real, y_real, frame]
						A[col_offset, row_offset] += Uconv[frame, i_basis]

			print(("\r      %04d/%04d                                         " % (i_basis, self.n_basis)), end='')
			sys.stdout.flush()
		print("")


		print("    computing gram matrix")
		b = np.asarray(np.dot(A.T, b).todense()).reshape(-1)
		A = np.asarray(np.dot(A.T, A).todense())

		print("    finding nnls solution")
		#A += np.diag(np.ones(A.shape[0])) * 1e-12
		x2 = np.linalg.lstsq(A, b)[0]
		x2[np.where(x2 < 0)] = 0
		#x = opt.nnls(A, b)[0]

		flt = np.zeros((self.filter_space.shape[1], self.filter_space.shape[2]))
		for i_filter in range(self.n_filter_space):
			flt = x2[offset:offset+flt_len].reshape((self.filter_space.shape[1], self.filter_space.shape[2]))
			norm = np.linalg.norm(flt)
			if norm == 0:
				print("    reinitializing filter %d" % i_filter)
				flt = self.init_flt_space(self.filter_space.shape[1], self.filter_space.shape[2])
			else:
				flt /= norm
			self.filter_space[i_filter, :, :] = flt

			import vigra
			vigra.impex.writeImage(np.asarray(flt), "flt_%d_%d.png" % (self.current_iteration ,i_filter))

			offset += flt_len


	def learn_positions(self):
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Uconv = self._fc_Uconv()
		cdef unsigned int i_filter
		cdef unsigned int i_basis
		cdef unsigned int j_basis
		cdef unsigned int maxidx_basis
		cdef unsigned int maxidx_filter

		tmp = np.sum(np.power(Uconv, 2), axis = 0)
		tmp = np.sqrt(tmp)
		tmp = np.tile(tmp, (Uconv.shape[0], 1))
		Uconv = np.divide(Uconv, tmp)
		Uconv[np.where(np.isnan(Uconv))] = 0

		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] UtU = np.dot(Uconv.T, Uconv)
		UtU[np.where(np.isnan(UtU))] = 0

		self.D_flt[...] = -1

		grad_shape = (self.n_basis, self.n_filter_space, self.n_rows, self.n_cols)
		cdef np.ndarray[SSTD_FLOAT_t, ndim=4] grad = np.zeros(grad_shape, dtype=SSTD_FLOAT_np) 

		print("    computing initial inner product")
		sys.stdout.flush()

		if self.n_frames == 1:
			#print(Uconv[:, 0])
			X = np.dot(self.input_nobg, Uconv[:, 0]).reshape((self.n_rows, self.n_cols))
			for i_filter in range(self.n_filter_space):
				grad[0, i_filter, :, :] = sig.correlate2d(X, self.filter_space[i_filter], mode='same', boundary='symm')
			for i_basis in range(1, self.n_basis):
				grad[i_basis, :, :, :] = grad[0, :, :, :]
		else:
			print(("      %04d/%04d" % (0, self.n_basis)), end='')
			for i_basis in range(self.n_basis):
				X = np.dot(self.input_nobg, Uconv[:, i_basis]).reshape((self.n_rows, self.n_cols))
				for i_filter in range(self.n_filter_space):
					grad[i_basis, i_filter, :, :] = sig.correlate2d(X, self.filter_space[i_filter], mode='same', boundary='symm')
					#print np.unique(grad)
				print("\r      %04d/%04d" % (i_basis, self.n_basis), end='')
				sys.stdout.flush()
			print("")

		cdef SSTD_UINT_t[:] avail = np.zeros(self.n_basis, dtype=SSTD_UINT_np)
		avail[:] = 1

		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] tmpimg = np.zeros((self.n_rows, self.n_cols))

		for i_basis in range(self.n_basis):
			maxidx_basis, maxidx_filter, maxidx_rows, maxidx_cols = np.unravel_index(np.argmax(grad), grad_shape)
			max_ = grad[maxidx_basis, maxidx_filter, maxidx_rows, maxidx_cols]

			print("    found cell basis %d filter %d at (%d,%d): %d" % (maxidx_basis, maxidx_filter, maxidx_rows, maxidx_cols, max_))
			if max_ <= 0:
				break

			assert(avail[maxidx_basis] == 1)
			avail[maxidx_basis] = 0

			self.D_flt[maxidx_basis] = maxidx_filter
			self.D_x[maxidx_basis] = maxidx_rows
			self.D_y[maxidx_basis] = maxidx_cols
			self.D_val[maxidx_basis] = max_

			tmpimg[...] = 0
			tmpimg[maxidx_rows, maxidx_cols] = max_

			print("      updating residual + dot product")
			grad[maxidx_basis, :, :, :] = np.NINF
			for i_filter in range(self.n_filter_space):
				tmpimg = sig.convolve2d(sig.convolve2d(tmpimg, self.filter_space[maxidx_filter, :, :], mode = 'same'), self.filter_space[i_filter, ::-1, ::-1], mode = 'same')
				for j_basis in range(self.n_basis):
					if avail[j_basis] == 0: continue
					grad[j_basis, i_filter, :, :] -= tmpimg * UtU[maxidx_basis, j_basis]
				

	def learn_spikes(self):
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Dconv = self._fc_Dconv()
		cdef np.ndarray tmp = np.sum(np.power(Dconv, 2), axis = 0)
		tmp = np.sqrt(tmp)
		tmp = np.tile(tmp, (Dconv.shape[0], 1))
		Dconv = np.divide(Dconv, tmp)
		Dconv[np.where(np.isnan(Dconv))] = 0

		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] DtD = np.dot(Dconv.T, Dconv)
		DtD[np.where(np.isnan(DtD))] = 0

		self.U_flt[:] = -1
		if self.n_frames == 1:
			for i_basis in range(self.n_basis):
				self.U_flt[i_basis] = 0
				self.U_basis[i_basis] = i_basis
				self.U_frame[i_basis] = 0
				self.U_val[i_basis] = 1
			return

		cdef unsigned int i_filter
		cdef unsigned int j_filter
		f_norm = []
		for i_filter in range(self.n_filter_time):
			norm = np.linalg.norm(self.filter_time[i_filter, :])
			f_norm.append(self.filter_time[i_filter, :]/norm)


		max_corr_len = 2*self.filter_time.shape[1]+1
		cdef np.ndarray[SSTD_FLOAT_t, ndim=3] flt_corr = np.zeros((self.n_filter_time, self.n_filter_time, max_corr_len), dtype=SSTD_FLOAT_np)
		for i_filter in range(self.n_filter_time):
			for j_filter in range(self.n_filter_time):
				flt_corr[i_filter, j_filter, :(len(f_norm[i_filter]) + len(f_norm[j_filter]) - 1)] = sig.correlate(f_norm[i_filter], f_norm[j_filter], mode = 'full')


		grad_shape = (self.n_basis, self.n_filter_time, self.n_frames)
		cdef np.ndarray[SSTD_FLOAT_t, ndim=3] grad_residual = np.zeros(grad_shape, dtype=SSTD_FLOAT_np)
		cdef np.ndarray basis_used = np.zeros(self.n_basis, dtype=np.int)
		for i_basis in range(self.n_basis):
			if np.sum(Dconv[:, i_basis]) > 0:
				basis_used[i_basis] = 1

			dataTDconv = np.asarray(np.dot(self.input_nobg.T, Dconv[:, i_basis]))
			for i_filter in range(self.n_filter_time):
				grad_residual[i_basis, i_filter, :] = sig.convolve(dataTDconv.reshape(-1), f_norm[i_filter].reshape(-1)[::-1], mode = 'full')[len(f_norm[i_filter])-1:][:grad_residual.shape[2]]

		cdef unsigned int maxidx_basis, maxidx_filter, maxidx_frame
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Ubasis = np.zeros((self.n_frames, self.n_filter_time), dtype=SSTD_FLOAT_np)
		diff = np.inf

		self.U_flt[:] = -1

		cdef unsigned int idx = 0
		cdef unsigned int i_iter
		for i_iter in range(self.learn_spikes_n_iter):
			Ubasis[:] = 0

			maxidx_basis, maxidx_filter, maxidx_frame = np.unravel_index(np.argmax(grad_residual), grad_shape)
			max_ = grad_residual[maxidx_basis, maxidx_filter, maxidx_frame]
			print ("    found spike for basis %d, filter %d at frame %d: %f" % (maxidx_basis, maxidx_filter, maxidx_frame, max_))
			if max_ <= 0.001:
				break

			Ubasis[maxidx_frame, maxidx_filter] += max_

			self.U_flt[idx] = maxidx_filter
			self.U_basis[idx] = maxidx_basis
			self.U_frame[idx] = maxidx_frame
			self.U_val[idx] = max_
			idx += 1

			for i_basis in range(self.n_basis):
				if basis_used[i_basis] == 0: continue

				for i_filter in range(self.n_filter_time):
					fltTflt = flt_corr[maxidx_filter, i_filter, :(len(f_norm[i_filter]) + len(f_norm[maxidx_filter]) - 1)]
					tmp2 = np.zeros(self.n_frames + len(fltTflt) - 1)
					tmp2[maxidx_frame:maxidx_frame + len(fltTflt)] = fltTflt * max_
					tmp2 = tmp2[len(f_norm[maxidx_filter]):][:grad_residual.shape[2]]
					grad_residual[i_basis, i_filter, :] -= tmp2 * DtD[maxidx_basis, i_basis]


	def learn_bg(self):
		if not self.do_bg:
			self.input_nobg = self.input_vanilla
			return

		if self.current_iteration % 10 == 0:
			self.did_bg = False

		if self.did_bg:
			return

		self.input_nobg[:,:] = estimate_bg(self.input_vanilla, self.n_rows, self.n_cols)
		self.did_bg = True

		import vigra
		tmp = np.array(self.input_nobg).reshape(self.n_rows, self.n_cols, self.n_frames)
		vigra.impex.writeImage(np.average(tmp, axis = 2), "avg_nobg.png")

	def run_single(self):
		print("  background estimation")
		self.learn_bg()

		if self.current_iteration != 0:
			print("  filter space")
			self.learn_filter_space()
			print("  filter time")
			self.learn_filter_time()

		print("  cell positions")
		self.learn_positions()
		print("  cell spikes")
		self.learn_spikes()

		self.current_iteration += 1

	def run(self, n_iter):
		for i in range(n_iter):
			self.run_single()


	def get_D(self):
		D = np.zeros((self.n_basis, self.n_filter_space, self.n_rows, self.n_cols))

		for i_basis in range(self.n_basis):
			if self.D_flt[i_basis] < 0: continue

			flt_idx = self.D_flt[i_basis]
			flt_x = self.D_x[i_basis]
			flt_y = self.D_y[i_basis]
			flt_val = self.D_val[i_basis]

			D[i_basis, flt_idx, flt_x, flt_y] = flt_val

		return D

	def get_U(self):
		U = np.zeros((self.n_basis, self.n_filter_time, self.n_frames))

		for i_spike in range(self.n_max_spikes):
			if self.U_flt[i_spike] < 0: break

			basis = self.U_basis[i_spike]
			flt = self.U_flt[i_spike]
			frame = self.U_frame[i_spike]

			U[basis, flt, frame] += self.U_val[i_spike]

		return U

	def get_H(self):
		return np.array(self.filter_space.copy())

	def get_f(self):
		return np.array(self.filter_time.copy())

	# internal helper
	def _fc_space(self, unsigned int idx):
		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] flt = np.zeros((self.filter_space.shape[1], self.filter_space.shape[2]), dtype=SSTD_FLOAT_np)

		flt_idx = self.D_flt[idx]
		flt_x = self.D_x[idx]
		flt_y = self.D_y[idx]
		flt_val = self.D_val[idx]


		flt[:] = self.filter_space[flt_idx, :, :]

		x_start = flt_x - (flt.shape[0] - 1) / 2
		y_start = flt_y - (flt.shape[1] - 1) / 2

		if x_start < 0:
			cut = -x_start
			flt = flt[cut:, :]
			x_start = 0

		if y_start < 0:
			cut = -y_start
			flt = flt[:, cut:]
			y_start = 0

		if x_start + flt.shape[0] > self.n_rows:
			cut = self.n_rows - x_start
			flt = flt[:cut, :]

		if y_start + flt.shape[1] > self.n_cols:
			cut = self.n_cols - y_start
			flt = flt[:, :cut]

		flt *= flt_val
		return (x_start, y_start, flt)

	def _reconstruct(self):
		cdef unsigned int i_basis
		cdef unsigned int i_spike
		cdef unsigned int time_flt_idx

		cdef np.ndarray dvol = np.zeros_like(self.input_vanilla)
		cdef np.ndarray[SSTD_FLOAT_t, ndim=1] time_flt = np.zeros(self.filter_time.shape[1], dtype=SSTD_FLOAT_np)

		for i_basis in range(self.n_basis):
			if self.U_flt[i_basis] < 0:
				continue

			x_start, y_start, flt = self._fc_space(i_basis)
			x_len, y_len = flt.shape

			st_evol = dvol[x_start:x_start+x_len, y_start:y_start+y_len, :]

			for i_spike in range(self.n_max_spikes):
				if self.U_basis[i_spike] < 0: break
				if self.U_basis[i_spike] != i_basis: continue

				time_flt_idx = self.U_flt[i_spike]
				time_flt[:] = self.filter_time[time_flt_idx, :].copy()
				time_flt *= self.U_val[i_spike]

				time_frame = self.U_frame[i_spike]

				evol = np.tile(flt, (len(time_flt),1)).reshape((len(time_flt), flt.shape[0], flt.shape[1]))
				tev = np.repeat(time_flt, np.prod(flt.shape)).reshape(evol.shape)
				evol = evol*tev
				evol = evol.swapaxes(0,2).swapaxes(0,1)

				st_evol_slice = st_evol[:,:,time_frame:time_frame+len(time_flt)]
				st_evol_slice += evol[:,:,:st_evol_slice.shape[2]]

		return dvol.reshape((self.n_rows*self.n_cols, self.n_frames))

	def _fc_Uconv(self):
		cdef unsigned int i_basis
		cdef unsigned int i_spike
		cdef unsigned int time_flt_idx

		cdef np.ndarray[SSTD_FLOAT_t, ndim=2] Uconv = np.zeros((self.n_frames, self.n_basis), dtype=SSTD_FLOAT_np)
		cdef np.ndarray[SSTD_FLOAT_t, ndim=1] time_flt = np.zeros(self.filter_time.shape[1], dtype=SSTD_FLOAT_np)

		if self.n_frames == 1:
			Uconv[0,:] = 1
			return Uconv

		for i_spike in range(self.n_max_spikes):
			if self.U_flt[i_spike] < 0: break

			time_basis = self.U_basis[i_spike]
			time_flt_idx = self.U_flt[i_spike]
			time_flt[:] = self.filter_time[time_flt_idx, :]
			time_flt *= self.U_val[i_spike]

			time_frame = self.U_frame[i_spike]
			time_flt_len = len(time_flt)

			slice_ = Uconv[time_frame:time_frame+time_flt_len, time_basis]
			slice_ += time_flt[:len(slice_)]


		return Uconv

	def _fc_Dconv(self):
		cdef np.ndarray[SSTD_FLOAT_t, ndim=3] Dconv = np.zeros((self.n_rows, self.n_cols, self.n_basis), dtype=SSTD_FLOAT_np)
		cdef unsigned int i_basis

		for i_basis in range(self.n_basis):
			if self.D_flt[i_basis] < 0: continue

			(x_start, y_start, flt) = self._fc_space(i_basis)
			x_len, y_len = flt.shape

			Dconv[x_start:x_start+x_len, y_start:y_start+y_len, i_basis] = flt

		return Dconv.reshape((self.n_rows*self.n_cols, self.n_basis))
