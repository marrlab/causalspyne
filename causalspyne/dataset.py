import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from causalspyne.dag_interface import MatDAG
from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.data_gen import DataGen


def simpson(size_sample=200):
    # 0 as confounder: 0->1, 0->2, 1->2
    mat_weighted_adjacency = np.array(
        [
            # 0 1 2 3
            [0, 0, 0],  # 0: confounder, root variable
            [3, 0, 0],  # 1: 0->1
            [-5, 1, 0],  # 2: 0->2, 1->2
        ]
    )

    dag = MatDAG(mat_weighted_adjacency,
                 name_prefix="V",
                 rng=default_rng())

    confounder = Idiosyncratic(class_name="Bernoulli",
                               dict_params={"p": 0.2},
                               rng=default_rng()
                               )

    data_gen = DataGen(dag, edge_model=None,
                       idiosynchratic={0: confounder})

    arr = data_gen.gen(size_sample)
    return arr


def visualize(arr):
    x = arr[:, 1]  # 2nd column
    y = arr[:, 2]  # 3rd column
    scenario = arr[:, 0]  # 1st column for scenario

    median0 = np.median(x[scenario == 0])
    median1 = np.median(x[scenario == 1])

    # Create treatment column: 1 if x > median for that z group, else 0
    treatment = np.zeros_like(x, dtype=int)
    treatment[(scenario == 0) & (x > median0)] = 1
    treatment[(scenario == 1) & (x > median1)] = 1
    # arr_discrete_aug = np.column_stack((arr, treatment))

    colors = ['green', 'orange']
    cmap = mcolors.ListedColormap(colors)
    # Boundaries separate the two values: 0 and 1
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    marker_map = {0: 'o', 1: 's'}

    for ind_val in np.unique(scenario):
        idx = np.where(scenario == ind_val)
        scatter = plt.scatter(x[idx], y[idx], c=treatment[idx],
                              marker=marker_map[ind_val],
                              edgecolor='k',
                              label=f'scenario {ind_val}',
                              cmap=cmap, norm=norm, s=100)

    plt.xlabel('treatment')
    plt.ylabel('performance')
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[0, 1])
    cbar.ax.set_yticklabels(['algorithm 1', 'algorithm 2'])
    plt.legend(title='scenario')
    plt.title('simpson treatment effect')
    return plt
