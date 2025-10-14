from causalspyne.dataset import simpson
import matplotlib.pyplot as plt

arr = simpson(200)

x = arr[:, 1]  # 2nd column
y = arr[:, 2]  # 3rd column
color = arr[:, 0]  # 1st column for color

plt.scatter(x, y, c=color, cmap='viridis')
plt.xlabel('2nd Column')
plt.ylabel('3rd Column')
plt.colorbar(label='1st Column')
plt.title('Scatter plot of 2nd vs 3rd column colored by 1st column')
plt.show()
