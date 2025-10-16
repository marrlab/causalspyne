import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import pandas as pd

from causalspyne.dag_interface import MatDAG
from causalspyne.noise_idiosyncratic import Idiosyncratic
from causalspyne.data_gen import DataGen


def simpson(size_sample=200, p=0.2,
            confounder_effect: float = -5,
            treatment_effect: float = 0.1,
            propensity: float = 3,
            std: float = 1.5):
    # 0 as confounder: 0->1, 0->2, 1->2
    mat_weighted_adjacency = np.array(
        [
            # 0 1 2 3
            [0, 0, 0],  # V0: confounder, root variable
            [propensity, 0, 0],  # V1: V0->V1, propensity of getting treatment
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
                       dict_params={"std": std},
                       idiosynchratic={0: confounder})

    arr = data_gen.gen(size_sample)
    scenario = arr[:, 0]  # 1st column for scenario/confounder
    treatment = arr[:, 1]  # 2nd column: treatment
    effect = arr[:, 2]  # 3rd column: effect/performance
    return scenario, treatment, effect


def tablize_simpson(scenario, treatment, effect, cut_off=0.75):
    y = effect
    effect = (y - np.min(y)) / (np.max(y) - np.min(y))
    if cut_off > 0:
        discrete_treatment = convert2discrete(scenario, treatment, cut_off)
    else:
        discrete_treatment = treatment
    arr_discrete_aug = np.column_stack((scenario, discrete_treatment, effect))
    return arr_discrete_aug


def convert2discrete(scenario, x, cut_off):
    ints_scenarios = np.unique(scenario)

    median_treatment = np.quantile(x, cut_off)

    discrete_treatment = np.zeros_like(x, dtype=int)  # discrete_treatment 0
    discrete_treatment[
        (scenario == ints_scenarios[0]) & (x > median_treatment)] = 1
    discrete_treatment[
        (scenario == ints_scenarios[1]) & (x > median_treatment)] = 1
    return discrete_treatment


def visualize_simpson(scenario, treatment, effect,
                      na_treatment="algorithm", na_confounder="scenario",
                      cut_off=0.75):
    x = treatment
    y = effect
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    ints_scenarios = np.unique(scenario)

    discrete_treatment = convert2discrete(scenario, x, cut_off)

    ints_treatment = np.unique(discrete_treatment)

    colors = ['green', 'orange']
    cmap = mcolors.ListedColormap(colors)
    # Boundaries separate the two values: 0 and 1
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    marker_map = {ints_scenarios[0]: 'o', ints_scenarios[1]: 's'}

    # fig, axs = plt.subplots(2, 2); ax=axs[0]
    fig, ax = plt.subplots()
    for ind_scenario in ints_scenarios:
        idx = np.where(scenario == ind_scenario)
        scatter = ax.scatter(
            x[idx], y[idx], c=discrete_treatment[idx],
            marker=marker_map[ind_scenario],
            edgecolor='k',
            label=f'scenario {ind_scenario}',
            cmap=cmap, norm=norm, s=100)

    ax.set_xlabel(f'jittered {na_treatment} w.r.t. {na_confounder}')
    ax.set_ylabel('performance')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=True)
    cbar = fig.colorbar(scatter, boundaries=bounds,
                        ticks=[ints_treatment[0], ints_treatment[1]])
    cbar.ax.set_yticklabels([f'{na_treatment} {ints_treatment[0]}',
                             f'{na_treatment} {ints_treatment[1]}'])
    proxy_o = Line2D([0], [0], marker='o', color='black', linestyle='None',
                     markerfacecolor='none')
    proxy_s = Line2D([0], [0], marker='s', color='black', linestyle='None',
                     markerfacecolor='none')
    ax.legend(title=f'{na_confounder}', handles=[proxy_o, proxy_s],
              labels=[f'{na_confounder} {ints_scenarios[0]}',
                      f'{na_confounder} {ints_scenarios[1]}'])
    ax.set_title("jittered scatter plot")
    fig.savefig("simpson_jitter.pdf")

    fig, axs = plt.subplots(2, 2)
    grouped_data = [y[discrete_treatment == g] for g in ints_treatment]
    axs[0, 1].boxplot(grouped_data, labels=ints_treatment)
    axs[0, 1].set_title(f"{na_confounder}s combined")
    axs[0, 1].set_xlabel(f'{na_treatment}')
    axs[0, 1].set_ylabel('performance')
    y0 = y[scenario == ints_scenarios[0]]
    discrete_treatment0 = discrete_treatment[scenario == ints_scenarios[0]]
    grouped_data0 = [y0[discrete_treatment0 == g] for g in ints_treatment]
    axs[1, 0].boxplot(grouped_data0, labels=ints_treatment)
    axs[1, 0].set_title(f"{na_confounder} {ints_scenarios[0]}")
    axs[1, 0].set_xlabel(f'{na_treatment}')
    axs[1, 0].set_ylabel('performance')

    y1 = y[scenario == ints_scenarios[1]]
    discrete_treatment1 = discrete_treatment[scenario == ints_scenarios[1]]
    grouped_data1 = [y1[discrete_treatment1 == g] for g in ints_treatment]
    axs[1, 1].boxplot(grouped_data1, labels=ints_treatment)
    axs[1, 1].set_title(f"{na_confounder} {ints_scenarios[1]}")
    axs[1, 1].set_xlabel(f'{na_treatment}')
    axs[1, 1].set_ylabel('performance')

    fig.suptitle('simpson treatment effect')
    fig.tight_layout()
    return fig


def adjust_simpson(scenario, treatment, effect):
    tab = tablize_simpson(scenario, treatment, effect)
    df = pd.DataFrame(tab, columns=["scenario", "treatment", "effect"])
    df_sampled = df.groupby(['scenario', 'treatment']).apply(
        lambda x: x.sample(n=3, random_state=42)).reset_index(drop=True)
    to_latex(df_sampled, "dataste", index=False)

    df_scenario = df['scenario'].value_counts(normalize=True).rename("p")
    to_latex(df_scenario, "scenarioPrior")

    df_treatment = df['treatment'].value_counts(normalize=True)
    to_latex(df_treatment, "treatmentMarginal")

    df_t_s = df.groupby('scenario')['treatment'].value_counts(
        normalize=True).rename('p(treatment | scenario)')
    to_latex(df_t_s, "treatmentScenario")

    df_s_t = df.groupby('treatment')['scenario'].value_counts(
        normalize=True).rename('p(scenario | treatment)')
    to_latex(df_s_t, "scenarioTreatment")

    df_e_st = df.groupby(['scenario', 'treatment'])['effect'].mean().rename(
        'E(effect | scenario, treatment)')
    to_latex(df_e_st, "effectScenarioTreatment")
    return df_scenario


def to_latex(df, na, index=True):
    latex_table = df.to_latex(
        index=index,
        buf=na + ".tex",
        caption=na,
        label="tab:mean_example",
        column_format="l l r",
        float_format="{:0.2f}".format,
        escape=False)
    return latex_table
