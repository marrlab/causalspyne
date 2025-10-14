from causalspyne.dataset import simpson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

arr = simpson(200)

x = arr[:, 1]  # 2nd column
y = arr[:, 2]  # 3rd column
color = arr[:, 0]  # 1st column for color

colors = ['green', 'orange']
cmap = mcolors.ListedColormap(colors)
# Boundaries separate the two values: 0 and 1
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
scatter = plt.scatter(x, y, c=color, cmap=cmap, norm=norm, s=100)

plt.xlabel('treatment')
plt.ylabel('performance')
# plt.colorbar(label='scenario')
cbar = plt.colorbar(scatter, boundaries=bounds, ticks=[0, 1])
cbar.ax.set_yticklabels(['Scenario 1', 'Scenario 2'])

plt.title('treatment effect')
plt.show()
