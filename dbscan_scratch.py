import numpy as np

class DB_SCAN:
	def __init__(self, eps, min ):
		self.eps = eps
		self.min = min
		self.labels_ = 0

	def fit(self, data):
		self.S = data

		"""
		Cluster a Dataset S using a manual implementation of the dbscan algorithm

		Parameters:
		`self.S`		- The dataset, list of all Datapoints
		`self.eps`		- Threshold distance
		`self.min` 		- Minimum required number of neighbors
		"""

		# x is the array that holds the Cluster values for each Datapoint, x[i] = label for S[i]
		# x[i] = -1 indicates unvisited datapoints
		# x[i] = 0 indicates noise
		# x[i] > 0 indicates the cluster
		# Mark all x[i] ∈ S as unvisited
		x = [-1]*len(self.S)

		# C is the value of the current cluster, starts at 1 (no noise)
		C = 1

		# For each point in the Dataset S
		for index in range(len(self.S)):
			# if x[i] is unvisited then
			if (x[index] == -1):

				# N ← neigh(x-i , ε)
				# Find all of S[index] neighboring points.
				N = self.neigh(index)

				# If the number is below min, this point is noise.
				if len(N) < self.min:
					# Mark x_i as noise;
					x[index] = 0

				# If there are at least min nearby, use this point as the core for a new cluster.
				else:
					#mark x_i as core
					x[index] = C

					#look for all other points in this cluster
					self.expand(x, N, C)

					#change to next cluster
					C += 1

		self.labels_=x

		#return

	def expand(self,x,N,C):
		"""
		Find all points in S that belong to a new cluster C.

		Parameters:
		`x`				- lables for Dataset S
		`N`				- All of the neighbors of `index`
		`C`				- The label for this new cluster
		"""

		j = 0
		while j < len(N):

			z = N[j]

			# if z is not visited
			if x[z] == -1:
				# Add z to cluster C (Assign cluster label C).
				x[z] = C

				# Find all the neighbors of z
				N_prime = self.neigh(z)

				# If Pn has at least min neighbors, it's a branch point --> update the neighbors
				if len(N_prime) >= self.min:
					N = N + N_prime

			# if z is not in any cluster, respectivley noise
			if x[z] == 0:
				# Add z to cluster C (Assign cluster label C).
				x[z] = C
				# rim???

			j += 1

	def neigh(self,index):
		"""
		Find all neigh points of S[index] in dataset S within distance self.eps

		Parameters:
		`index			- index number of the datapoint in the Dataset S
		"""

		neighbors = []

		# For each point in the dataset...
		for p in range(len(self.S)):

			# If the distance is below the threshold, add it to the neighbors list.
			if np.linalg.norm(self.S[index] - self.S[p]) < self.eps:
				neighbors.append(p)

		return neighbors
