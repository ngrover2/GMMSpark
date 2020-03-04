import numpy as np
from GMMUtils import FLOAT_EPSILON as EPS
import math

class MultiVariateGaussian:

	def __init__(self, meanV, covM):
		self.mean = meanV
		self.sigma = covM
		self.evals, self.evecs = np.linalg.eigh(covM) # Eigendecomposition(sigma) = evecs * diag(evals) * evecs.T
		self._cutoff = self._not_zero_cutoff(self.evals)
		self.pdf_mahalanobis_sigma_inverse_part = self._mahalanobis_sigma_inverse_part()
		self.pdf_constant_part = self._constant_part()
		

	def _mahalanobis_sigma_inverse_part(self):
		# Mahalanobis distance involves computing inverse of sigma. We use eigendecomposition of sigma to calculate Mahalanobis distance
		""" 
		Step 1 : 
		Take the square root of eigenvalues vector and diagonalise the resulting vector. 
		That would be the inverse square root of the diagonal matrix in the eigendecomposition of sigma.
		"""

		inv_sqrt_fn = np.vectorize(lambda x: np.sqrt( 1 / x ) if x > self._cutoff else 0)
		inv_sqrt_diag = np.diag(inv_sqrt_fn(self.evals))
		""" 
		Step 2 : 
		Then take the dot product of inv_sqrt_diag with eigenvalues matrix's transpose 
		"""
		return np.dot(inv_sqrt_diag, self.evecs.T)
	
	def _constant_part(self):
		"""
			This calculates the log of pdf's constant term at the beginning i.e.
			log( (2*pi)^(-k/2)^ * det(sigma)^(-1/2)^ ) , where k is the dimension of the data
		"""
		log_pseudo_det_sigma = np.sum(np.array(list(map(lambda x: math.log(x), filter(lambda x: x > self._cutoff, self.evals)))))
		return -0.5 * ( (len(self.mean) * np.log(2 * math.pi)) + log_pseudo_det_sigma )


	def pdf(self, X):
		return np.exp(self.log_pdf(X))

	def log_pdf(self, X):
		dist_from_mean = (X - self.mean)
		mahaMulDist = np.dot(self.pdf_mahalanobis_sigma_inverse_part,  dist_from_mean)
		return self.pdf_constant_part + (-0.5 * (np.dot(mahaMulDist.T, mahaMulDist)))
		
	@staticmethod
	def _not_zero_cutoff(V):
		VdtypeChar = V.dtype.char.lower()
		return np.max(V) * (1E3 if VdtypeChar == 'f' else 1E6) * np.finfo(VdtypeChar).eps
	