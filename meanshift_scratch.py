import numpy as np
import math

class Mean_Shift:
	def __init__(self, bandwidth = None, norm_step = 0.5, max_iter = 50):
		self.bandwidth = bandwidth
		self.norm_step = norm_step
		self.max_iter = max_iter

	def neighbour_points(self, centroid, distance = 5):
		return [x for x in self.data if np.linalg.norm(x- centroid) <= distance]

	def gaussian_kernel(self, distance, bandwidth):
		return (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)

	def fit(self,data):
		#bandwidth estimation
		if self.bandwidth == None:
			#find the center of the entire data
			all_data_centroid = np.average(data,axis=0)
			#magnitude of the average
			all_data_norm = np.linalg.norm(all_data_centroid)
			#determine an overall bandwidth
			self.bandwidth = all_data_norm/self.norm_step

		#Start: Every datapoint as a cluster center
		self.data = data
		centroids = np.copy(data)

		total_iter =0
		while 1:
			new_centroids = []

			#loop over known centroids
			for i,c in enumerate(centroids):

				neighbours = self.neighbour_points(c, self.bandwidth*3)

				num, den = 0, 0
				#iterate over neighbours wether or not feature is in bandwitdh
				for neighbour in neighbours:
					#eucl. distance
					distance = np.linalg.norm(neighbour-c)
					#gaussian distributed wheigts according to distance
					weight = self.gaussian_kernel(distance,self.bandwidth)
					num += (weight * neighbour)
					den += weight

				#mean vector of all vectors
				new_centroid = num / den

				#add to centroids list
				new_centroids.append(new_centroid)

			# look for very similar new_centroids (costly: O(n^2) time)
			duplicates = []
			for i in range(len(new_centroids)):
				for j in range(1, len(new_centroids)):
					if i != j:
						if np.linalg.norm(new_centroids[j] - new_centroids[i]) < 0.01:
							duplicates.append(j)

			# remove near duplicates
			new_centroids = [i for j, i in enumerate(new_centroids) if j not in duplicates]

			#check if centroids are moving
			if len(new_centroids) == len(centroids):
				error = np.linalg.norm(centroids - new_centroids, 1)
				if error <= 0.01:
					print("centers converged")
					break
				elif total_iter > self.max_iter:
					print("reached max iterations")
					break

			centroids = np.copy(new_centroids)
			total_iter+=1

		self.centroids = centroids

		self.labels_= np.zeros(len(data))

		for i,x in enumerate(data):
			d = [np.linalg.norm(x - c, axis=0) for c in self.centroids]
			self.labels_[i] = np.argmin(d)
