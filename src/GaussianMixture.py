import numpy as np
from pyspark import RDD
from MultiVariateGaussian import MultiVariateGaussian as MVG
from pprint import pprint
import traceback
from functools import partial
import ujson
import sys
import os

class GaussianMixture:

	def __init__(self,sc, num_gaussians=2, min_llh_change=0.1, num_iterations=10, seed=12345, init_gauss_num_samples=10):
		self.k = num_gaussians
		self.min_llh_change = min_llh_change
		self.num_iterations = num_iterations
		self.random_seed = seed
		self.num_samples = init_gauss_num_samples
		self.sc = sc
	
	def calculate_vector_mean(self, vectors):
		return np.mean(vectors, axis=0)
	
	def create_covariance_matrix(self, vectors):
		mu = self.calculate_vector_mean(vectors)
		dims = mu.shape[0]
		variance = np.zeros(dims)
		diffMeanVector = np.zeros(dims)
		for vec in vectors:
			variance += np.power((vec - mu), 2)
		num_samples = len(vectors)
		if num_samples > 0:
			variance /= num_samples
		return np.diag(variance)
	
	def mix(self, word_vectors_rdd):
		#  word_vectors_rdd -> RDD OF WORD VECTORS
		sc = self.sc

		data_rdd = word_vectors_rdd.map(lambda x: x.split(",")[1:]) # x.split(",")[0] is the word, it is not needed for GMM run, so it is discarded
		# data_vectors_rdd = data_rdd.map(lambda x: np.array(x)).map(lambda x: x.astype(np.float)).map(lambda x: x/np.linalg.norm(x,2)).cache() # some suggest normalisation, I dint see any improvement
		data_vectors_rdd = data_rdd.map(lambda x: np.array(x)).map(lambda x: x.astype(np.float)).cache()
		data_vectors_rdd = data_vectors_rdd.cache()

		num_dimensions = data_vectors_rdd.first().shape[0]
		try:
			gaussians = []
			weights = []
			N = data_vectors_rdd.count()

			# GMM is HIGHLY sensitive to initialisation, so manual initialisation did not work well
			# I use means initialised with Kmeans on the dataset. For large datasets, Kmeans can be run on a sample and used for Spark
			
			# This initialisation DOES NOT work. All Gaussians get concentrated in the centre
			
			# init_samples = data_vectors_rdd.takeSample(False, self.num_samples * self.k, self.random_seed)
			# for kidx in range(self.k):
			# 	samples_per_gaussian = init_samples[ kidx * self.num_samples : (kidx + 1) * self.num_samples ]
			# 	this_gauss_mean = self.calculate_vector_mean(samples_per_gaussian)
			# 	this_gauss_cov = self.create_covariance_matrix(samples_per_gaussian)
			# 	gaussians.append( MVG(this_gauss_mean, this_gauss_cov))
			# 	weights.append( 1 / self.k ) # uniform weights
			
			
			# This initialisation works
			# use Kmeans output for initialisation of gaussians' means, I use 41 clusters for my dataset, as discovered by BIC score
			jsonpath = os.path.join(os.path.dirname(__file__), "initial_means_kmeans.json")
			kmeans_means = np.zeros((self.k, num_dimensions))
			with open(jsonpath, 'r') as kmeansfile:
				line_num = 0
				for line in kmeansfile:
					if line != "":
						d = ujson.loads(line)
						kmeans_means[line_num] = np.array(d[f"mean_{line_num}"])
						line_num += 1

			for kidx in range(self.k):
				this_gauss_mean = kmeans_means[kidx]
				this_gauss_cov = np.diag([10. for _ in range(num_dimensions)]).astype(np.float)
				gaussians.append( MVG(this_gauss_mean, this_gauss_cov))
				weights.append( 1 / self.k ) # uniform weights
			
			itZero_file = None
			try:
				itZero_file = open(f"iteration0_gmm_params.json",'w+')
				for idx, gaussian in enumerate(gaussians):
					per_gaussian_params = dict()
					per_gaussian_params['mean'] = gaussian.mean
					per_gaussian_params['covariance'] = gaussian.sigma
					json_string = ujson.dumps(per_gaussian_params)
					print(json_string ,file=itZero_file)
				itZero_file.close()
			except:
				pass
			
			lfile = None
			try:
				lfile = open("./log_likelihoods.txt", 'w+')
			except:
				pass
			
			# distribute the weights and gaussians so that their parameters can be updated for each iteration
			prev_log_likelihood = 0.0
			current_log_likelihood = 0.0
			
			for i in range(self.num_iterations):
				seqAdd = sc.broadcast( partial(GMMParameters.seqAdd, weights, gaussians ) )
				gmm_params = None
				gmm_params = data_vectors_rdd.treeAggregate(GMMParameters(self.k, num_dimensions), seqAdd.value, (lambda gmm_params1, gmm_params2: gmm_params1.combAdd( gmm_params2)) )
				# gmm_params hold parameters aggregated over all partitions. 
				# However, the means, sigmas and weights held in gmm_params are 
				# unnormalised as we did not have total number of data points when computing those
				# But now we do and we will normalise those values

				total_responsibility_mass = gmm_params.responsibilities.sum() # this represents mass of all points for all clusters i.e. the total mass. Individual cluster mass can be accessed through gmm_params.responsibilities[cluster index]
				# For each cluster we will normalise the values appropriately
				for cluster_idx in range(self.k):
					(new_weight, new_gaussian) = self.updateWeightsAndGaussians(lfile, gmm_params.responsibilities[cluster_idx], gmm_params.means[cluster_idx], gmm_params.covMatrices[cluster_idx], total_responsibility_mass, N)
					weights[cluster_idx] = new_weight
					gaussians[cluster_idx] = new_gaussian

				prev_log_likelihood = current_log_likelihood
				current_log_likelihood = gmm_params.log_likelihood
				# print(f"Iteration {i} log_likelihood:{current_log_likelihood} prev_log_likelihood:{prev_log_likelihood} \n")
				if abs(prev_log_likelihood - current_log_likelihood) < 0.5:
					break
				try:
					# for local testing
					lfile.write(f"Iteration {i} log_likelihood:{current_log_likelihood}\n")
					itfile = open(f"iteration{i+1}_gmm_params.json",'w+')
					for idx, gaussian in enumerate(gaussians):
						per_gaussian_params = dict()
						per_gaussian_params['mean'] = gaussian.mean
						per_gaussian_params['covariance'] = gaussian.sigma
						json_string = ujson.dumps(per_gaussian_params)
						print(json_string ,file=itfile)
					
					itfile.close()
				except:
					pass
				seqAdd.destroy()
			
			if data_vectors_rdd:
				data_vectors_rdd.unpersist()
			
			return GMMModel(weights, gaussians)
		except:
			print(traceback.format_exc())
			if data_vectors_rdd:
				data_vectors_rdd.unpersist()
			return None

	def updateWeightsAndGaussians(self, lfile, cluster_responsibilty_mass, unnormalised_cluster_mean, unnormalised_cluster_sigma, total_resp_mass, N):
		this_gauss_new_mean = unnormalised_cluster_mean / cluster_responsibilty_mass
		# this_gauss_new_weight = cluster_responsibilty_mass / total_resp_mass
		this_gauss_new_weight = cluster_responsibilty_mass / N
		
		# 1 trying with diagonal covariance below 
		# unnormalised_cluster_sigma += np.diag(-cluster_responsibilty_mass *  np.power(this_gauss_new_mean ,2))
		# return (this_gauss_new_weight, MVG(this_gauss_new_mean, unnormalised_cluster_sigma / cluster_responsibilty_mass))
		
		# 2 trying with full covariance below 
		this_gauss_new_mean_as_1byN = this_gauss_new_mean.reshape(1,-1)
		unnormalised_cluster_sigma += -cluster_responsibilty_mass *  np.dot(this_gauss_new_mean_as_1byN.T, this_gauss_new_mean_as_1byN)
		return (this_gauss_new_weight, MVG(this_gauss_new_mean, unnormalised_cluster_sigma / cluster_responsibilty_mass))
		
		
class GMMParameters:
	
	def __init__(self, num_clusters, num_dimensions):
		self.log_likelihood = 0.0
		self.num_clusters = num_clusters
		self.num_dimensions = num_dimensions
		self.responsibilities = np.zeros(num_clusters)
		self.means = [np.zeros(num_dimensions) for _ in range(num_clusters)]
		self.covMatrices = [np.zeros((num_dimensions, num_dimensions)) for _ in range(num_clusters)]
		self.eps = np.finfo('f').eps
	
	@classmethod
	def seqAdd( cls, weights, gaussians, existing_params, new_data_point ):
		
		responsibilities = np.array(list(map(lambda wght_gauss : np.finfo('f').eps + wght_gauss[0] * wght_gauss[1].pdf(new_data_point) , zip(weights, gaussians))))
		sum_responsibility = np.sum(responsibilities) # sum_responsibility is for the data point that it is computed for, it is not the total responsibility. We will need to collect all responsibility and then normalise the weights and means and sigmas. For now, collect the values for the numerator in the upadte equations for weights, means and sigmas 
		existing_params.log_likelihood += np.log(sum_responsibility)
		for k in range(len(weights)):
			responsibilities[k] /= sum_responsibility

			existing_params.responsibilities[k] += responsibilities[k]
			existing_params.means[k] += new_data_point * responsibilities[k]
			# 1 try diagonal covariance below
			# existing_params.covMatrices[k] += np.diag(responsibilities[k] * np.power( new_data_point, 2 )) # here we need to create the covariance matrix from the data point
			# 2 try full covariance below
			new_data_point_as_1byN = new_data_point.reshape(1,-1)
			existing_params.covMatrices[k] += (responsibilities[k] * np.dot( new_data_point_as_1byN.T, new_data_point_as_1byN )) # here we need to create the covariance matrix from the data point
		return existing_params
	
	def combAdd(self, new_params ):
		for k in range(self.num_clusters):
			self.responsibilities[k] += new_params.responsibilities[k]
			self.means[k] += new_params.means[k]
			self.covMatrices[k] += new_params.covMatrices[k]
			
		self.log_likelihood += new_params.log_likelihood
		return self

class GMMModel:
	def __init__(self, weights, gaussians):
		self.weights = weights
		self.gaussians = gaussians
		try:
			# for local testing
			for i in range(len(weights)):
				print(f"Cluster{i} weight -> {weights[i]}")
				print(f"Cluster{i} mean -> {gaussians[i].mean}")
				print(f"Cluster{i} sigma -> {gaussians[i].sigma}")
		except:
			pass
	
	def predict_soft(self, data_point):
		predictions = []
		for gaussian in self.gaussians:
			predictions.append(gaussian.pdf(data_point))
		return predictions
	
	def predict_hard(self, data_point):
		predictions = []
		for gaussian in self.gaussians:
			predictions.append(gaussian.pdf(data_point))
		return (np.argmax(np.array(predictions)), np.max(np.array(predictions)))
