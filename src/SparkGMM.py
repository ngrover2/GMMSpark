from pyspark import RDD, SparkContext, StorageLevel
from pyspark import SparkConf
import traceback
import sys
import os
from GaussianMixture import GaussianMixture as GM
import numpy as np
import ujson
import time


if __name__ == "__main__":
	sc = None
	try:
		try:
			# conf = SparkConf().setAll([
			# 		('spark.executor.cores', '3'),
			# 		('spark.executor.memory', '5g'),
			# 		('spark.yarn.excecutor.memoryOverhead', '1g'),
			# 		('spark.driver.cores', '1'),
			# 		('spark.executor.instances', '3'),
			# 		('spark.default.parallelism', '18'),
			# 	]).setAppName("Test_GMM")
			
			sc = SparkContext('local[4]')
			sc.addPyFile("./src/GaussianMixture.py")
			sc.addPyFile("./src/MultiVariateGaussian.py")
			sc.addPyFile("./src/GMMUtils.py")
			print("SparkContext Initialised!")
		except:
			print("Could not initialise SparkContext :(")
			print(traceback.format_exc())
			sys.exit(1)
		
		if len(sys.argv) < 3:
			print("""
			Extra Arguments missing: 
			arg1 : Input file that contains string word vectors of the form --> word,comma separated vector values ex: cloud,2.8976,3.40987,1.0222...
			arg2 : Number of clusters
			arg3 : Number of Iterations 
			arg4 : number of samples per gaussian
			arg5 : output directory for saving topic predictions
			""")
			sys.exit(1)
		
		arg_input_path = sys.argv[1]
		input_path = None
		if isinstance(arg_input_path, str) and len(arg_input_path) > 0:
			input_path = arg_input_path
		
		# Process argument 2
		arg_num_clusters = sys.argv[2]
		num_clusters = None
		try:
			num_clusters = int(arg_num_clusters)
		except:
			pass
		
		if num_clusters is None:
			num_clusters = 2
		
		# Process argument 3
		arg_num_iterations = sys.argv[3]
		num_iterations = None
		try:
			num_iterations = int(arg_num_iterations)
		except:
			pass
		
		if num_iterations is None or num_iterations > 100:
			num_iterations = 50
		
		# Process argument 4
		arg_num_samples = sys.argv[4]
		num_samples = None
		try:
			num_samples = int(arg_num_samples)
		except:
			pass
		
		if num_samples is None:
			num_samples = 10
		
		# Process argument 5
		
		arg_output_dir = sys.argv[5]
		output_dir = None
		if isinstance(arg_output_dir, str) and len(arg_output_dir) > 0:
			output_dir = arg_output_dir
		
		if output_dir is None:
			epoch_time = time.time()
			output_dir = f"./output{epoch_time}"
		
		if num_iterations is None or num_iterations > 100:
			num_iterations = 50

		# Report that the input path is empty and exit
		if not input_path:
			print("Input path (argument 1) is invalid")
			sys.exit(1)
		
		word_vectors_rdd = None
		try:
			print(f"Reading from {input_path}")
			word_vectors_rdd = sc.textFile(input_path)
		except:
			print("Input path is invalid OR does not contain a text file")
			print("RDD could not be created :(")
			sys.exit(1)
		
		word_vectors_rdd = word_vectors_rdd.cache()
		
		gm = GM(sc, num_gaussians=num_clusters, min_llh_change=0.1, num_iterations=num_iterations, seed=12345, init_gauss_num_samples = num_samples )
		
		model = gm.mix(word_vectors_rdd)

		brd_model = None
		single_topic_predictions = None
		word_vector_keypair_rdd = None
		# Topic mapping for each word as per the model
		try:
			if model is None:
				print("Gaussian Mixture Model could not be computed")
			else:
				brd_model = sc.broadcast(model)
				word_vector_keypair_rdd = word_vectors_rdd \
									.map(lambda x: (x.split(",")[0], np.array(x.split(",")[1:]).astype(np.float)) ).cache()

				word_vector_keypair_rdd \
					.map(lambda x: (x[0], brd_model.value.predict_soft(x[1]))) \
						.map(lambda x: f"{x[0]},{','.join([str(val) for val in x[1]])}") \
						.saveAsTextFile(os.path.join(output_dir, f"predictions_soft.txt"))
				
				single_topic_predictions = word_vector_keypair_rdd \
						.map(lambda x: (x[0], brd_model.value.predict_hard(x[1]))).cache()
				
				for cl in range(num_clusters):
					single_topic_predictions \
						.filter(lambda x: x[1][0] == cl) \
							.map(lambda x: f"{x[0]},{str(x[1][1])}") \
							.saveAsTextFile(os.path.join(output_dir, f"predictions_topic{cl}.txt"))
				
				with open(f"./gmm_model_params.json",'w+') as local_gmm_model_params_file:
					for idx, (weight, gaussian) in enumerate(zip(model.weights, model.gaussians)):
						per_gaussian_params = dict()
						per_gaussian_params['mean'] = gaussian.mean
						per_gaussian_params['covariance'] = gaussian.sigma
						json_string = ujson.dumps(per_gaussian_params)
						print(json_string ,file=local_gmm_model_params_file)

			
		except:
			print(traceback.format_exc())
		finally:
			if brd_model:
				brd_model.destroy()
			if word_vectors_rdd:
				word_vectors_rdd.unpersist()
			if word_vector_keypair_rdd:
				word_vector_keypair_rdd.unpersist()
			if single_topic_predictions:
				single_topic_predictions.unpersist()
			if sc:
				sc.stop()

	except SystemExit:
		pass
	except:
		print("Error Ocurred")
		print(traceback.format_exc())
		pass
	finally:
		if sc:
			sc.stop()
	
