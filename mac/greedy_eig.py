import numpy as np

import networkx as nx
import networkx.linalg as la

class GreedyEig:
    def __init__(self, odom_measurements, lc_measurements, num_poses):
        self.L_odom = weight_graph_lap_from_edge_list(odom_measurements, num_poses)
        self.num_poses = num_poses
        self.laplacian_e_list = []
        self.weights = []

        for meas in lc_measurements:
            laplacian_e = weight_graph_lap_from_edge_list([meas], num_poses)
            self.laplacian_e_list.append(laplacian_e)
            self.weights.append(meas.weight)
            self.edge_list.append((meas.i,meas.j))

        self.laplacian_e_list = np.array(self.laplacian_e_list)
        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

    def find_fiedler_pair(self, L, method='tracemin_lu', tol=1e-8):
        """
        Compute the second smallest eigenvalue of L and corresponding
        eigenvector using `method` and tolerance `tol`.

        w: An element of [0,1]^m; this is the edge selection to use
        method: Any method supported by NetworkX for computing algebraic
        connectivity. See:
        https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html

        tol: Numerical tolerance for eigenvalue computation

        returns a tuple (lambda_2(L), v_2(L)) containing the Fiedler
        value and corresponding vector.

        """
        assert(method != 'lobpcg') # LOBPCG not supported at the moment
        find_fiedler_func = la.algebraicconnectivity._get_fiedler_func(method)
        x = None
        output = find_fiedler_func(L, x=x, normalized=False, tol=tol, seed=np.random.RandomState(7))
        return output

    def combined_laplacian(self, w, tol=1e-10):
        """
        Construct the combined Laplacian (fixed edges plus candidate edges weighted by w).

        w: An element of [0,1]^m; this is the edge selection to use
        tol: Tolerance for edges that are numerically zero. This improves speed
        in situations where edges are not *exactly* zero, but close enough that
        they have almost no influence on the graph.

        returns the matrix L(w)
        """
        idx = np.where(w > tol)
        prod = w[idx]*self.weights[idx]
        C1 = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_poses)
        C = self.L_odom + C1
        return C

    def subset(self, k, save_intermediate=False):
        solution = np.zeros(len(self.weights))
        for i in range(k):
            # Placeholders to keep track of best measurement
            best_idx = -1
            best_l2 = 0
            # Loop over all unselected measurements to find new best
            # measurement to add
            for j in range(len(self.weights)):
                # If measurement j is already selected, skip it
                if solution[j] > 0:
                    continue
                # Test solution
                w = np.copy(solution)
                w[j] = 1
                L = self.combined_laplacian(w)
                l2 = self.find_fiedler_pair(L)[0]
                if l2 >= best_l2:
                    best_idx = j
                    best_l2 = l2
            # If best_idx is still -1, something went terribly wrong, or there
            # are no measurements
            assert(best_idx != -1)
            solution[best_idx] = 1
        return solution


