"""
A probabilistic optimal estimation method for detecting spatial fuzzy communities
"""
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import hdbscan
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

DEFAULT_NUM_REALIZATIONS = 20  # Default number of realizations
DEFAULT_MAX_ITER = 500  # Default maximum number of iterations
DEFAULT_MIN_ITER = 100  # Default minimum number of iterations
DEFAULT_MAX_K = 20  # Default maximum number of communities
DEFAULT_INF = 1e20  # Infinity
DEFAULT_EPS = 1e-20  # Used to prevent division by zero
DEFAULT_L_STEADY = 1e-3  # Threshold for log-likelihood stability
DEFAULT_U_MIN = 1e-5  # Prevents membership (probability) from being too small
DEFAULT_U_STEADY = 1e-7  # Threshold for membership stability
DEFAULT_U_MAX_ITER = 500  # Maximum number of iterations for membership
DEFAULT_ALPHA = 2  # Parameter for spatial constraint
DEFAULT_SPATIAL_CONTINUITY_METHOD = 'Adj'  # Spatial continuity method

class ProOE:
    def __init__(self, Es, Distances, Adj, Vs, Vs_coord, K, spatial_continuity_method = DEFAULT_SPATIAL_CONTINUITY_METHOD, alpha = DEFAULT_ALPHA, max_iter=DEFAULT_MAX_ITER, min_iter=DEFAULT_MIN_ITER, max_k=DEFAULT_MAX_K, num_realizations=DEFAULT_NUM_REALIZATIONS, seed=None, verbose=True):
        """
                Initialize the class,
        
                Parameters:
                Vs: Set of nodes (node vector file type: geodataframe including unit_id, geometry columns; unit_id is of text type; geometry is point)
                Vs_coord: Node coordinates (type: dictionary, key is unit_id, value is coordinate)
                Es: Set of edges (OD flow type: dataframe including O_id, D_id, flow; O_id, D_id correspond to unit_id in Vs, type is str; flows are OD flows)
                Ds: Distance between nodes (distance type: dataframe, including O_id, D_id, distance columns; O_id, D_id correspond to unit_id in Vs, type is str; distance is the distance between OD)
                K: Number of communities (number type: int, number of communities) If K is None, the number of communities is adaptively selected
                max_iter: Maximum number of iterations (number type: int)
                spatial_continuity_method: Spatial continuity method (string type: 'Adj' or 'HDBSCAN')
                min_iter: Minimum number of iterations (number type: int)
                max_k: Maximum number of communities (number type: int)
                alpha: Parameter for spatial constraint (number type: float)
                seed: Random seed (number type: int)
                verbose: Whether to print process information (bool type: True or False)
                num_realizations: Number of repetitions (number type: int)
        """
        self.verbose = verbose
        if self.verbose:
            print('Preparating...')
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.max_k = max_k  
        self.alpha = alpha 
        self.spatial_continuity_method = spatial_continuity_method

        if self.verbose:
            print('Fitting probability density function...')
        self.mobilityDistribution = MobilityDistribution(Distances, Es)
        if self.verbose:
            self.mobilityDistribution.print_params()
            self.mobilityDistribution.plot()
        
        if self.verbose:
            print('Calculating probability density...')
        dis_density = Distances.copy(deep=True)
        dis_density['data'] = self.mobilityDistribution.predict_PDF(dis_density['data'].values)
        
        if self.verbose:
            print('Data preparation is complete...')
        self.Vs = Vs  # Set of nodes
        self.Vs_coord = Vs_coord  # Node coordinates

        if self.verbose:
            print('Creating running instances...')
        self.k2best_I = {}  # Store the optimal community intensity array for each number of communities
        self.k2best_U = {}  # Store the optimal community membership matrix for each number of communities
        self.k2best_L = {}  # Store the optimal log-likelihood value for each number of communities

        # print('准备运行变量...')
        if self.verbose:
            print('Preparing running variables...')
        self.unit_id2index = {}
        for i in range(len(self.Vs)):
            self.unit_id2index[self.Vs[i]] = i

        self.Es = {(i, j): 0.0 for i in self.Vs for j in self.Vs if i != j}
        self.Ds = {(i, j): DEFAULT_INF for i in self.Vs for j in self.Vs if i != j}
        self.Adj = {(i, j): False for i in self.Vs for j in self.Vs if i != j}
        self.PDFs = {(i, j): DEFAULT_EPS for i in self.Vs for j in self.Vs if i != j}
        for index, row in Es.iterrows():
            self.Es[(row['O_id'], row['D_id'])] = row['data']
        for index, row in Distances.iterrows():
            self.Ds[(row['O_id'], row['D_id'])] = row['data']
        for index, row in Adj.iterrows():
            self.Adj[(row['O_id'], row['D_id'])] = row['data']
        for index, row in dis_density.iterrows():
            self.PDFs[(row['O_id'], row['D_id'])] = row['data'] if row['data'] > 0 else DEFAULT_EPS

        if isinstance(Vs, list):
            Vs = pd.DataFrame(Vs, columns=['unit_id'])
        ijdf = pd.DataFrame()
        ijdf['O_id'] = np.repeat(Vs['unit_id'].values, len(Vs['unit_id']))
        ijdf['D_id'] = np.tile(Vs['unit_id'].values, len(Vs['unit_id']))
        ijs = set([(i, j) for i, j in zip(ijdf['O_id'], ijdf['D_id']) if i != j])
        self.keys = list(ijs)
        self.keys_i = np.array([self.unit_id2index[key[0]] for key in self.keys])
        self.keys_j = np.array([self.unit_id2index[key[1]] for key in self.keys])
        self.PDFs_keys = np.array([self.PDFs[key] for key in self.keys])
        self.build_matrices() 

        self.init_seed(seed)  

        if K is not None and K <= 0:
            raise ValueError('K should be greater than 0')
        elif K is None:
            print('Adaptive selection of the number of communities...')
            knee = self.adaptive_K()
            self.K = knee
            self.Ks = np.arange(knee) 
            self.num_realizations = num_realizations
            if self.verbose:
                print('Start fitting...')
            self.fit()
        else:
            self.K = K
            self.Ks = np.arange(K) 
            self.num_realizations = num_realizations
            if self.verbose:
                print('Start fitting...')
            self.fit()
        
        # print('准备结束')
        if self.verbose:
            print('Preparation is over')
        
    
    def adaptive_K(self):
        lls4ks = {}
        # Step 1: Calculate the log-likelihood value for each number of communities
        for k in range(1, self.max_k + 1):
            print(f'  --Fitting the number of communities: {k}')
            self.K = k
            self.Ks = np.arange(k)  # Array of community numbers
            self.num_realizations = 3
            self.fit()
            lls4ks[k] = self.k2best_L[k]
    
        # Step 2: Store the log-likelihood values as an array, prepare for fitting
        k_range = np.array(list(lls4ks.keys()))
        lls = np.array(list(lls4ks.values()))
    
        # Define a custom exponential function
        def diy_func(x, a, b, c):
            return a * np.log(b * x + DEFAULT_EPS) + c
    
        # Step 3: Polynomial fitting (choose an appropriate polynomial degree)
        poly_coeffs = np.polyfit(k_range, lls, deg=4)  # Polynomial fitting
        poly = np.poly1d(poly_coeffs)
        lls_poly = poly(k_range)
    
        # Step 4: Use the kneed library to find the knee point
        kneedle = KneeLocator(k_range, lls_poly, curve='convex', direction='increasing')
        steady_point = kneedle.knee
        

        print('The number of communities obtained adaptively is:', steady_point)
    
        # Step 5: Plot the smoothed curve and its steady point
        if self.verbose:
            plt.plot(k_range, lls, label='Original Loglikelihood')
            plt.plot(k_range, lls_poly, label='Smoothed Loglikelihood', linestyle='--')
            plt.xlabel('Number of communities')
            plt.ylabel('Loglikelihood')
            plt.title('Loglikelihood curve (smoothed)')
            if steady_point is not None:
                plt.axvline(steady_point, color='green', linestyle='--', label=f'Steady point at {steady_point}')
            plt.legend()
            plt.show()
    
        return steady_point

    def build_matrices(self):
        num_nodes = len(self.Vs)
        self.Es_matrix = np.zeros((num_nodes, num_nodes))
        self.Ds_matrix = np.full((num_nodes, num_nodes), DEFAULT_INF)
        self.Adj_matrix = np.zeros((num_nodes, num_nodes))
        self.PDFs_matrix = np.full((num_nodes, num_nodes), DEFAULT_EPS)

        Es_indices = np.array([self.unit_id2index[i] for i, j in self.Es.keys()]), np.array([self.unit_id2index[j] for i, j in self.Es.keys()])
        Es_values = np.array(list(self.Es.values()))
        self.Es_matrix[Es_indices] = Es_values
    
        Ds_indices = np.array([self.unit_id2index[i] for i, j in self.Ds.keys()]), np.array([self.unit_id2index[j] for i, j in self.Ds.keys()])
        Ds_values = np.array(list(self.Ds.values()))
        self.Ds_matrix[Ds_indices] = Ds_values
    
        Adj_indices = np.array([self.unit_id2index[i] for i, j in self.Adj.keys()]), np.array([self.unit_id2index[j] for i, j in self.Adj.keys()])
        Adj_values = np.array(list(self.Adj.values()))
        self.Adj_matrix[Adj_indices] = Adj_values
    
        PDF_indices = np.array([self.unit_id2index[i] for i, j in self.PDFs.keys()]), np.array([self.unit_id2index[j] for i, j in self.PDFs.keys()])
        PDF_values = np.array(list(self.PDFs.values()))
        self.PDFs_matrix[PDF_indices] = PDF_values

    def loglikelihood(self, I, U):
        """
            Calculate the log-likelihood value
    
            Parameters:
            I: Community intensity (type: list, each value corresponds to the intensity of a different community)
            U: Community membership (type: matrix, each row is a node, each column is a community)
    
            Returns:
            L: Log-likelihood value (type: float)
        """
        LE = self.calcu_LE(I, U)
        return LE
    
    def calcu_LE(self, I, U):
        """
                Calculate the log-likelihood value of the interaction part
        """
        lambda_ij = np.einsum('k,ik,jk,ij->ij', I, U, U, self.PDFs_matrix)
        LE = np.sum(self.Es_matrix * np.log(lambda_ij + DEFAULT_EPS) - lambda_ij)
        return LE
    
    def init_seed(self, seed):
        """
                Initialize the random seed
        """
        if seed is None:
            seed = int(time.time())
        self.prng = np.random.RandomState(seed)

    def init_I(self):
            """
            Initialize the community intensity array
            """
            I = self.prng.rand(self.K)
            return I
    
    def init_U(self):
        """
        Initialize the community membership matrix
        """
        U = self.prng.rand(len(self.Vs), self.K)
        U = self.constraint_U(U)
        return U

    def constraint_U(self, U):
        U = np.where(np.isnan(U) | (U < DEFAULT_U_MIN), 0, U)
        row_sums = U.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        U = U / row_sums
        return U

    def constraint_spatial(self, U):
        """
        Spatial constraint
        """
        if self.alpha == 0:
            return U
        elif self.spatial_continuity_method == 'Adj':
            return self.constraint_spatial_Adj(U)
        elif self.spatial_continuity_method == 'HDBSCAN':
            return self.constraint_spatial_HDBSCAN(U)

    def constraint_spatial_HDBSCAN(self, U):
        """
                Spatial constraint
                Traverse each type of community
                For each type of community, extract all nodes belonging to that community
                Cluster these nodes based on the HDBSCAN algorithm
                Compare the mean probability of belonging to the community for each cluster, and select the cluster with the highest mean probability
                Set the probability of nodes not belonging to this cluster to zero for that community
        """
        for k in range(self.K):
            U_k = U[:, k]
            U_k_non_zero = U_k[U_k > 0.1]
            U_k_non_zero_index = np.where(U_k > 0.1)[0]
            if len(U_k_non_zero) <= 2:
                continue
            # Extract the corresponding coordinates from self.Vs_coord
            coord = [self.Vs_coord[i] for i in U_k_non_zero_index]
            # Use hdbscan for clustering
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
            clusterer.fit(coord)
            # Extract clustering results
            labels = clusterer.labels_

            # Extract the number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # print(f"k:{k+1} ,n_clusters:{n_clusters},set(labels):{set(labels)}")
            if n_clusters == 0:
                continue

            # Calculate the sum of U for each cluster
            cluster_sums = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_sums[i] = np.sum(U_k_non_zero[labels == i])
            # Extract the cluster with the largest sum
            max_cluster = np.argmax(cluster_sums)
            # Reduce the membership of nodes not belonging to this cluster
            U_k[U_k_non_zero_index[labels != max_cluster]] = U_k[U_k_non_zero_index[labels != max_cluster]] / self.alpha
            U[:, k] = U_k
        U = self.constraint_U(U)
        return U

    def constraint_spatial_Adj(self, U):
        """
        Spatial constraint, use geographic adjacency constrains to adjust the membership matrix U
        Traverse each community
        For each community, extract all nodes belonging to that community with membership greater than 0.2
        Generate all connected graphs based on the adjacency matrix
        Compare the sum of membership for each connected graph, and select the connected graph with the highest sum
        The membership of nodes not belonging to this connected graph is divided by alpha
        """
        for k in range(self.K):
            U_k = U[:, k]
            U_k_non_zero = U_k[U_k > 0.1]
            U_k_non_zero_index = np.where(U_k > 0.1)[0]
            if len(U_k_non_zero) <= 2:
                continue
            # Generate all connected graphs based on the adjacency matrix
            connected_graphs = self.generate_connected_graphs(U_k_non_zero_index)

            # Calculate the sum of U for each connected graph
            graph_sums = np.zeros(len(connected_graphs))
            for i in range(len(connected_graphs)):
                # get the indexs of the connected graph in the U_k_non_zero_index
                connected_graph = connected_graphs[i]
                # 将connected_graph中的值，与U_k_non_zero_index中的值对应起来，转为U_k_non_zero_index的索引位置值
                connected_graph_index = [np.where(U_k_non_zero_index == connected_graph[j])[0][0] for j in range(len(connected_graph))]
                graph_sums[i] = np.sum(U_k_non_zero[connected_graph_index])
                
            # Extract the connected graph with the largest sum
            max_graph = np.argmax(graph_sums)
            # Reduce the membership of nodes not belonging to this connected graph
            U_k[U_k_non_zero_index[~np.isin(U_k_non_zero_index, connected_graphs[max_graph])]] = U_k[U_k_non_zero_index[~np.isin(U_k_non_zero_index, connected_graphs[max_graph])]] / self.alpha
            U[:, k] = U_k
        U = self.constraint_U(U)
        return U

    def generate_connected_graphs(self, U_k_non_zero_index):
        """
        Generate all connected graphs based on the adjacency matrix
        """
        connected_graphs = []
        visited = np.zeros(len(U_k_non_zero_index))
        for i in range(len(U_k_non_zero_index)):
            if visited[i] == 0:
                connected_graph = self.dfs(U_k_non_zero_index, visited, i)
                connected_graphs.append(connected_graph)
        return connected_graphs
    
    def dfs(self, U_k_non_zero_index, visited, i):
        """
        Depth-first search
        """
        connected_graph = []
        stack = [i]
        while stack:
            node = stack.pop()
            if visited[node] == 0:
                visited[node] = 1
                connected_graph.append(U_k_non_zero_index[node])
                for j in range(len(U_k_non_zero_index)):
                    if self.Adj_matrix[U_k_non_zero_index[node], U_k_non_zero_index[j]] == 1 and visited[j] == 0:
                        stack.append(j)
        return connected_graph

    def fit(self):
        """
        Fit U and I in the model using the EM algorithm
        """

        for _ in range(self.num_realizations):
            self.realization = _ + 1
            I, U = None, None
            I, U, L = self.fit_EM(I, U)
            
            L = self.loglikelihood(I, U)
            # print(f"\nThe {_+1}th fitting ended, the log-likelihood value is: {L}")
            if L > self.k2best_L.get(self.K, -DEFAULT_INF):
                # print(colored(f"The optimal Loglikelihood value of the {_+1} fitting is: {L}, which is the current Optimal Log_Likelihood value", 'green'))
                print(colored(f"The optimal Loglikelihood value of the {_+1} fitting is: {L}, which is the current Optimal Log_Likelihood value", 'green'))
                self.k2best_I[self.K] = I
                self.k2best_U[self.K] = U
                self.k2best_L[self.K] = L
            else:
                print(' ')

        # Sort the data in self.k2best_U[self.K] and self.k2best_I[self.K] according to self.k2best_I[self.K],
        # and store the sorted data into self.k2best_U[self.K] and self.k2best_I[self.K]
        self.k2best_I[self.K], self.k2best_U[self.K] = self.sort_TU(self.k2best_I[self.K], self.k2best_U[self.K])
    
    def sort_TU(self, I, U):
        """
        Sort I and U
        """
        # Sort I and record the sorted indices
        I_sort_index = np.argsort(I)[::-1]
        # Sort the columns of U according to the indices
        U_sort = U[:, I_sort_index]

        return I[I_sort_index], U_sort
            
    def fit_EM(self, I = None, U = None):
        """
        Fit I and U using the EM algorithm
        """
        if I is None:
            I = self.init_I()
        if U is None:
            U = self.init_U()

        best_I = I
        best_U = U
        best_L = -DEFAULT_INF

        # Iteration process
        self.start_time = time.time()
        ls = []
        for _ in range(self.max_iter):

            q = self.update_q(I, U)
            U = self.update_U(I, U, q)
            U = self.constraint_spatial(U)   # Apply spatial constraint if _ is a multiple of 10
            q = self.update_q(I, U)
            I = self.update_T(U, q)

            # Calculate log-likelihood value
            L = self.loglikelihood(I, U)
            print(colored(f"\r    Realization{self.realization}/{self.num_realizations}", 'yellow'), end="")
            print(f" iter:{_+1}/{self.max_iter}, loglikelihood:{L:8.3f}, time:{(time.time()-self.start_time):5.2f}s ", end="")
            ls.append(L)
            if len(ls) > self.min_iter and (np.isnan(L) or np.std(ls[-10:]) < DEFAULT_L_STEADY or L < ls[-10]):
                break

            if L > best_L:
                best_L = L
                best_I = I
                best_U = U
        return best_I, best_U, best_L


    def update_q(self, I, U):
        """
        Update q
        """
        
        q = np.einsum('k,ik,jk,ij->ijk', I, U, U, self.PDFs_matrix)
        row_sums = np.sum(q, axis=2)
        q = q / (row_sums[:,:, np.newaxis] + DEFAULT_EPS)
        
        return q
    
    def update_T(self, U, q_matrix):
        """
        Update I
        """
        I = np.zeros(self.K)

        numerators = np.sum(self.Es_matrix[:, :, np.newaxis] * q_matrix, axis=(0, 1))  # Calculate the numerator
        
        # Calculate U_i and U_j matrices
        U_i_matrix = U[self.keys_i, :]  # (num_edges, num_features)
        U_j_matrix = U[self.keys_j, :]  # (num_edges, num_features)
        
        dominators = np.sum(U_i_matrix * U_j_matrix * self.PDFs_keys[:, np.newaxis], axis=0)  # Calculate the denominator

        I = (numerators + DEFAULT_EPS) / (dominators + DEFAULT_EPS)

        return I

    def update_U(self, I, U, q_matrix):
        """
        Update U
        The following will traverse all nodes and update them based on their type
        """

        for i in self.prng.permutation(self.Vs):
            U = self.update_U_interaction(I, U, q_matrix, i)
            U = self.constraint_U(U)
        
        return U

    def update_U_interaction(self, I, U, q_matrix, i):
        """
        Update interaction node i
        """

        i_index = self.unit_id2index[i]
        j_indices = np.where(self.Es_matrix[i_index, :] > 0)[0]  # Select nodes related to i with interactions

        Es_i_js = self.Es_matrix[i_index, j_indices]
        q_i_js = q_matrix[i_index, j_indices]

        numerator_ks = np.sum(Es_i_js[:, np.newaxis] * q_i_js[:, self.Ks], axis=0)

        j_indices = np.where(self.Es_matrix[i_index, :] >= 0)[0]  # Select nodes related to i
        j_indices = j_indices[j_indices != i_index]

        pdfs = self.PDFs_matrix[i_index, j_indices]
        U_PDFs_ks = U[j_indices][:,self.Ks] * pdfs[:, np.newaxis]
        dominator_ks = I[self.Ks] * np.sum(U_PDFs_ks, axis=0)

        U[i_index][self.Ks] = numerator_ks / dominator_ks

        return U

    
    def normal_Non_zero(self, x):
        # Find the boolean index of elements greater than 0
        idx = x > 0
        
        # Calculate the sum of each row, only summing elements greater than 0
        row_sums = np.sum(x * idx, axis=1, keepdims=True)
        
        # Replace 0 in row_sums with 1 to avoid errors, does not affect the result
        row_sums[row_sums == 0] = 1
        
        # Normalize each row
        x = x / row_sums
        
        # Restore the original 0 elements
        x[~idx] = 0
        
        return x

    def deduplication(self, df):
        """
        Remove duplicates, rows in the DataFrame are undirected but saved twice in the DataFrame, remove duplicates here

        Parameters:
        df: Original DataFrame

        Returns:
        new_df: DataFrame after removing duplicates
        """
        
        # Create a set to store unique edges
        unique_edges = set()
        
        def edge_key(row):
            """Helper function to create a unique key for each edge."""
            o_id, d_id = row['O_id'], row['D_id']
            # Ensure the order of undirected edges is consistent
            return tuple(sorted((o_id, d_id)))
        
        # Filter unique edges
        mask = df.apply(lambda row: edge_key(row) not in unique_edges and not unique_edges.add(edge_key(row)), axis=1)
        
        # Use boolean mask to filter out unique rows
        new_df = df[mask].reset_index(drop=True)
        
        return new_df

    def save_result(self, spatialunit, id_name,  outputpath, name_suffix=''):
        """
        Save results
        """

        # Check if the path exists, if not, create it
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        
        # Save community intensity I
        I_df = pd.DataFrame(self.k2best_I[self.K])
        I_df.columns = ['I']
        # Add a column 'c' representing the community, corresponding to 'k' below
        I_df['c'] = [f'c_{k+1}' for k in range(self.K)]

        I_df.to_csv(outputpath + 'I' + name_suffix + '.csv', index=False)

        # Save log-likelihood value L
        L_value = self.k2best_L[self.K]
        with open(outputpath + 'L' + name_suffix + '.csv', 'w') as f:
            f.write(str(L_value))

        # Save community membership U
        # Remove columns other than id_name from spatialunit
        spatialunit = spatialunit[[id_name, 'geometry']].copy()
        # Add self.K columns to spatialunit, each representing a community
        for k in range(self.K):
            spatialunit.loc[:, f'c_{k+1}'] = 0.0

        # Iterate over spatialunit and fill values from self.k2best_U into spatialunit
        for index, row in spatialunit.iterrows():
            unit_id = row[id_name]
            if unit_id not in self.unit_id2index:
                continue
            U = self.k2best_U[self.K][self.unit_id2index[unit_id]]
            for k in range(self.K):
                spatialunit.loc[index, f'c_{k+1}'] = U[k]
        # Save spatialunit as a shp file
        # spatialunit.to_file(outputpath + 'U' + name_suffix + '.shp', encoding='utf-8')
        # Save spatialunit as a geojson file
        spatialunit.to_file(outputpath + 'U' + name_suffix + '.geojson', driver='GeoJSON', encoding='utf-8')

        # Calculate the corresponding q for the optimal I and U
        q = self.update_q(self.k2best_I[self.K], self.k2best_U[self.K])
        # Save q as a DataFrame with columns O_id, D_id, O_lng, O_lat, D_lng, D_lat, c_1, c_2, ..., c_K
        q_list = []
        for i in range(len(self.Vs)):
            for j in range(i, len(self.Vs)):
                temp = {'O_id': self.Vs[i], 'D_id': self.Vs[j], 'O_lng': self.Vs_coord[i][0], 'O_lat': self.Vs_coord[i][1], 'D_lng': self.Vs_coord[j][0], 'D_lat': self.Vs_coord[j][1]}
                for k in range(self.K):
                    temp[f'c_{k+1}'] = q[i, j, k]
                q_list.append(temp)
        q_df = pd.DataFrame(q_list)
        q_df.to_csv(outputpath + 'q' + name_suffix + '.csv', index=False)

        if self.verbose:
            print('\nResults saved successfully!')
    
    def get_best_result(self):
        """
        Get the best result
        """
        return self.k2best_I[self.K], self.k2best_U[self.K], self.k2best_L[self.K]

    def construct_Expected_Es(self, outputpath, name_suffix=''):
        """
        Construct the expected interaction table
        """
        Expected_Es_matrix = np.zeros((len(self.Vs), len(self.Vs)))
        for i in range(len(self.Vs)):
            for j in range(i, len(self.Vs)):
                if i == j:
                    Expected_Es_matrix[i, j] = 0
                else:
                    Expected_Es_matrix[i, j] = np.sum(self.k2best_I[self.K] * self.k2best_U[self.K][i] * self.k2best_U[self.K][j] * self.PDFs_matrix[i, j])
                    Expected_Es_matrix[j, i] = Expected_Es_matrix[i, j]
        
        # Convert the matrix to DataFrame
        Expected_Es_list = []
        for i in range(len(self.Vs)):
            for j in range(i, len(self.Vs)):
                Expected_Es_list.append({'O_id': self.Vs[i], 'D_id': self.Vs[j], 'data': Expected_Es_matrix[i, j]})
        Expected_Es = pd.DataFrame(Expected_Es_list)
        
        # Ensure self.Vs_coord is a DataFrame
        if isinstance(self.Vs_coord, list):
            self.Vs_coord = pd.DataFrame(self.Vs_coord)
        
        # Combine self.Vs and self.Vs_coord to construct the point coordinate table, self.Vs_coord only includes coordinate information
        Vs_coord = self.Vs_coord.copy()
        Vs_coord.columns = ['lng', 'lat']
        Vs_coord['id'] = self.Vs
        Vs_coord = Vs_coord[['id', 'lng', 'lat']]
        
        # Merge the interaction table with the point coordinate table
        Expected_Es = Expected_Es.merge(Vs_coord, left_on='O_id', right_on='id', how='left', suffixes=('', '_o'))
        Expected_Es = Expected_Es.merge(Vs_coord, left_on='D_id', right_on='id', how='left', suffixes=('', '_d'))
        Expected_Es.columns = ['O_id', 'D_id', 'data', 'id_o',  'lng_o', 'lat_o', 'id_d', 'lng_d', 'lat_d']
        Expected_Es = Expected_Es.drop(columns=['id_o', 'id_d'])
    
        # Save the expected interaction table
        Expected_Es.to_csv(outputpath + 'Expected_Es' + name_suffix + '.csv', index=False)
    
        return Expected_Es


class MobilityDistribution:
    def __init__(self, dis, inter):
        self.interaction = dis.copy(deep=True)
        merged_df = self.interaction.merge(inter[['O_id', 'D_id', 'data']], on=['O_id', 'D_id'], how='left', suffixes=('', '_new'))
        self.interaction['data'] = merged_df['data_new'].combine_first(self.interaction['data'])
        self.interaction = self.interaction['data'].values
        self.distance = dis['data'].values
        self.upper_bound = self.distance.max()
        self.lower_bound = 0.0
        self.params = None
        self.binnums = 50    
        self.bins = np.linspace(self.lower_bound, self.upper_bound, self.binnums)
        self.bin_mids = 0.5 * (self.bins[:-1] + self.bins[1:])
        
        # Build data for fitting
        self.data = self._construct_data()
        # Fit both distributions and select the best one
        self.fit()

    def _construct_data(self):
        distances = np.array(self.distance)
        interactions = np.array(self.interaction)
        interacttimes = np.zeros(self.binnums - 1)
        for i in range(len(self.bins) - 1):
            interacttimes[i] = interactions[(distances > self.bins[i]) & (distances < self.bins[i + 1])].sum()
        data = interacttimes / interacttimes.sum()
        interacttimes = interacttimes.astype(int)
        self.fitdata = np.repeat(self.bin_mids, interacttimes)
        return data

    def _fit_lognormal(self, valid_data):
        # Use original data for fitting
        shape, loc, scale = stats.lognorm.fit(valid_data, floc=0)
        # Calculate predicted values and normalize
        y_pred = stats.lognorm.pdf(self.bin_mids, shape, loc, scale)
        y_pred = y_pred / (np.sum(y_pred) * (self.bins[1] - self.bins[0]))
        r2 = self._calculate_r2(self.data, y_pred)
        return (shape, loc, scale), r2, 'lognormal'

    def _fit_powerlaw(self, valid_data):
        # Use original data for fitting
        shape, loc, scale  = stats.powerlaw.fit(valid_data, floc=0)
        # Calculate predicted values and normalize
        y_pred = stats.powerlaw.pdf(self.bin_mids, shape, loc, scale)
        y_pred = y_pred / (np.sum(y_pred) * (self.bins[1] - self.bins[0]))
        r2 = self._calculate_r2(self.data, y_pred)
        return (shape, loc, scale), r2, 'powerlaw'


    def _calculate_r2(self, y_true, y_pred):

        # Scale y_true and y_pred to a uniform scale based on maximum values
        y_true = y_true / np.max(y_true)
        y_pred = y_pred / np.max(y_pred)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + DEFAULT_EPS))
        return r2

    def fit(self):
        valid_data = self.fitdata[self.fitdata > 0]
        if len(valid_data) == 0:
            raise ValueError("All distance values are non-positive after filtering.")

        # Fit both distributions
        self.lognorm_params, self.lognorm_r2, _ = self._fit_lognormal(valid_data)
        self.powerlaw_params, self.powerlaw_r2, _ = self._fit_powerlaw(valid_data)

        # Select the distribution with higher R²
        if self.lognorm_r2 > self.powerlaw_r2:
            self.params = self.lognorm_params
            self.distribution = 'lognormal'
            self.r2 = self.lognorm_r2
        else:
            self.params = self.powerlaw_params
            self.distribution = 'powerlaw'
            self.r2 = self.powerlaw_r2
        print(f"Selected distribution: {self.distribution} (R² = {self.r2:.4f})")

    def print_params(self):
        print(f"Lognormal distribution (R² = {self.lognorm_r2:.4f})")
        print(f"Parameters: shape={self.lognorm_params[0]:.4f}, loc={self.lognorm_params[1]:.4f}, scale={self.lognorm_params[2]:.4f}")
        print(f"\nPower law distribution (R² = {self.powerlaw_r2:.4f})")
        print(f"Parameters: alpha={self.powerlaw_params[0]:.4f}, A={self.powerlaw_params[1]:.4f}")
        print(f"\nSelected distribution: {self.distribution}")

    def predict_PDF(self, x, dist_type=None):
        if dist_type is None:
            dist_type = self.distribution
            
        if dist_type == 'lognormal':
            shape, loc, scale = self.lognorm_params
            y = stats.lognorm.pdf(x, shape, loc, scale)
        else:
            shape, loc, scale = self.params
            y = stats.powerlaw.pdf(x, shape, loc, scale)
        return y
    
    def plot(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
        # Plot observed data
        ax1.scatter(self.bin_mids, self.data, alpha=0.5, label='Observed Data', color='black')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Frequency')
    
        # Create shared x-axis
        ax2 = ax1.twinx()
        
        # Plot only the better-fitting distribution
        distances = np.linspace(self.lower_bound + DEFAULT_EPS, self.upper_bound, 1000)
        
        # Get PDF for selected distribution
        pdf = self.predict_PDF(distances)
        if self.distribution == 'lognormal':
            ax2.plot(distances, pdf, 'r-', 
                    label=f'Lognormal (R² = {self.r2:.4f})')
        else:
            ax2.plot(distances, pdf, 'b--', 
                    label=f'Power Law (R² = {self.r2:.4f})')
        
        ax2.set_ylabel('Probability Density')   
    
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
        plt.title(f'Distance Distribution Fitting: {self.distribution.capitalize()}')
        fig.tight_layout()
        plt.show()
