import matplotlib.pyplot as plt
from causalspyne.dataset import simpson, visualize
arr = simpson(200)

plot = visualize(arr)
plot.savefig('simpson_scatter_plot.pdf', format='pdf')
plt.show()
