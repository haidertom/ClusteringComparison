import numpy as np
import random

class K_Means:
	def __init__(self, k=2, tol = 0.001):
		self.k = k
		self.tol = tol

	def fit(self,data):

		#Randomly select k points c_0,..., c_k−1 from S. These are the centers of the clusters.
		self.centroids = np.array([data[i] for i in random.sample(range(0, len(data)), self.k)])
		self.labels_ = np.zeros(len(data))

		C_old = np.zeros(self.centroids.shape)
		while 1:
			#For each xi ∈ S, assign xi to that cluster the center of which is closest.
			for i,x in enumerate(data):
				d = [np.linalg.norm(x - c, axis=0) for c in self.centroids]
				self.labels_[i] = np.argmin(d)

			#deep copy of old values, otherwise its just a pointer
			C_old = np.copy(self.centroids)

			#Re-compute the centers cj to be the centroids of Cj.
			for i in range(self.k):
				points = [data[j] for j in range(len(data)) if self.labels_[j] == i]
				self.centroids[i] = np.mean(points, axis=0)

			error = np.linalg.norm(self.centroids - C_old, 1)
			if error <= self.tol:
				break
