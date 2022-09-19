import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_measurements, split_measurements, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
from mac.greedy_eig import GreedyEig
from mac.mac import MAC

plt.rcParams['text.usetex'] = True

G = nx.petersen_graph()
n = len(G.nodes())

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

print(G)
pos = nx.shell_layout(G, nlist=[range(5,10), range(5)])
nx.draw(G, pos=pos)
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

measurements = nx_to_mac(G)

# Split chain and non-chain parts
fixed_meas, candidate_meas = split_measurements(measurements)

pct_candidates = 0.3
num_candidates = int(pct_candidates * len(candidate_meas))
mac = MAC(fixed_meas, candidate_meas, n)
greedy_eig = GreedyEig(fixed_meas, candidate_meas, n)

w_init = np.zeros(len(candidate_meas))
w_init[:num_candidates] = 1.0

result, unrounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=50)
greedy_eig_result = greedy_eig.subset(num_candidates)

init_selected = select_measurements(candidate_meas, w_init)
greedy_eig_selected = select_measurements(candidate_meas, greedy_eig_result)
selected = select_measurements(candidate_meas, result)

init_selected_G = mac_to_nx(fixed_meas + init_selected)
greedy_eig_selected_G = mac_to_nx(fixed_meas + greedy_eig_selected)
selected_G = mac_to_nx(fixed_meas + selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(141)
nx.draw(G, pos=pos)
plt.title(rf"Original ($\lambda_2$ = {mac.evaluate_objective(np.ones(len(w_init))):.3f})")
plt.subplot(142)
nx.draw(init_selected_G, pos=pos)
plt.title(rf"Naive ($\lambda_2$ = {mac.evaluate_objective(w_init):.3f})")
plt.subplot(143)
nx.draw(greedy_eig_selected_G, pos=pos)
plt.title(rf"GreedyEig ($\lambda_2$ = {mac.evaluate_objective(greedy_eig_result):.3f})")
plt.subplot(144)
nx.draw(selected_G, pos=pos)
plt.title(rf"Ours ($\lambda_2$ = {mac.evaluate_objective(result):.3f})")
plt.show()

