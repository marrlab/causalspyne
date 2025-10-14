import numpy as np
from causalspyne.dataset import simpson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

arr = simpson(200)

x = arr[:, 1]  # 2nd column
y = arr[:, 2]  # 3rd column
color = arr[:, 0]  # 1st column for color

median0 = np.median(x[color == 0])
median1 = np.median(x[color == 1])

# Create indicator column: 1 if x > median for that z group, else 0
indicator = np.zeros_like(x, dtype=int)
indicator[(color == 0) & (x > median0)] = 1
indicator[(color == 1) & (x > median1)] = 1
arr_discrete_aug = np.column_stack((arr, indicator))

colors = ['green', 'orange']
cmap = mcolors.ListedColormap(colors)
# Boundaries separate the two values: 0 and 1
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
marker_map = {0: 'o', 1: 's'}

for ind_val in np.unique(indicator):
    idx = np.where(indicator == ind_val)
    scatter = plt.scatter(x[idx], y[idx], c=color[idx],
                          marker=marker_map[ind_val],
                          edgecolor='k',
                          label=f'algo {ind_val}',
                          cmap=cmap, norm=norm, s=100)

plt.xlabel('treatment')
plt.ylabel('performance')
# plt.colorbar(label='scenario')
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[0, 1])
cbar.ax.set_yticklabels(['Scenario 1', 'Scenario 2'])
plt.legend(title='algorithm')
plt.title('treatment effect')
plt.show()
