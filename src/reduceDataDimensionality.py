from MulticoreTSNE import MulticoreTSNE as TSNE
import os
import numpy as np
import sys

if __name__ == "__main__":
	try:
		args = sys.argv
		if len(args) < 2:
			print("Arguments missing: Expected two arguments -> arg1: input data file path that holds the vectors and arg2: output file path")
			raise SystemExit
		# print(args)
		data_file_path = os.path.join( os.path.dirname(__file__), args[1])
		output_file_path = os.path.join( os.path.dirname(__file__), args[2])

		list_vectors = []
		list_words = []
		with open(data_file_path,'r') as dfile:
			print(f"Reading from {data_file_path}..")
			line_num = 0
			for line in dfile:
				line_num += 1
				split = line.split(',')
				word_part = split[0]
				list_words.append(word_part)
				vec_part = split[1:]
				extracted_vec = list(map(lambda x: float(x), vec_part))
				list_vectors.append(extracted_vec)
		
		print("Loading vectors for TSNE..")
		X = np.array(list_vectors)

		tsne = TSNE(n_jobs=4)

		print("Performing TSNE..")
		Y = tsne.fit_transform(X)

		print("Saving new vectors to file..")
		with open(output_file_path, 'w+') as wfile:
			for idx, new_vec in enumerate(Y):
				print(f"{list_words[idx]},{','.join([str(x) for x in new_vec])}", file = wfile)
			print(f"Saved TSNEed vectors at {output_file_path}.")

	except SystemExit:
		pass
	except:
		raise



