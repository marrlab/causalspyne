import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from causalspyne.dag_interface import MatDAG
from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.data_gen import DataGen


def simpson(size_sample=200, p=0.2,
            confounder_effect: float = -5,
            treatment_effect: float = 1,
            propensity: float = 3):
    # 0 as confounder: 0->1, 0->2, 1->2
    mat_weighted_adjacency = np.array(
        [
            # 0 1 2 3
            [0, 0, 0],  # 0: confounder, root variable
            [propensity, 0, 0],  # 1: 0->1
            [confounder_effect, treatment_effect, 0],  # 2: 0->2, 1->2
        ]
    )

    dag = MatDAG(mat_weighted_adjacency,
                 name_prefix="V",
                 rng=default_rng())

    confounder = Idiosyncratic(class_name="Bernoulli",
                               dict_params={"p": p},
                               rng=default_rng()
                               )

    data_gen = DataGen(dag, edge_model=None,
                       idiosynchratic={0: confounder})

    arr = data_gen.gen(size_sample)
    return arr


def visualize(arr, na_treatment="algorithm", na_confounder="scenario"):
    scenario = arr[:, 0]  # 1st column for scenario/confounder
    x = arr[:, 1]  # 2nd column: treatment
    y = arr[:, 2]  # 3rd column: effect/performance

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

    for ind_scenario in np.unique(scenario):
        idx = np.where(scenario == ind_scenario)
        scatter = plt.scatter(x[idx], y[idx], c=treatment[idx],
                              marker=marker_map[ind_scenario],
                              edgecolor='k',
                              label=f'scenario {ind_scenario}',
                              cmap=cmap, norm=norm, s=100)

    plt.xlabel(f'jittered {na_treatment} w.r.t. {na_confounder}')
    plt.ylabel('performance')
    plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='y', labelleft=False)
    cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[0, 1])
    cbar.ax.set_yticklabels([f'{na_treatment} 1', f'{na_treatment} 2'])
    proxy_o = Line2D([0], [0], marker='o', color='black', linestyle='None',
                     markerfacecolor='none')
    proxy_s = Line2D([0], [0], marker='s', color='black', linestyle='None',
                     markerfacecolor='none')
    plt.legend(title=f'{na_confounder}', handles=[proxy_o, proxy_s],
               labels=[f'{na_confounder} 1', f'{na_confounder} 2'])
    plt.title('simpson treatment effect')
    return plt
