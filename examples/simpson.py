import matplotlib.pyplot as plt
from causalspyne.dataset import simpson, visualize_simpson


scenario, treatment, effect = simpson(200)
plot = visualize_simpson(scenario, treatment, effect)
plot.savefig('simpson_scatter_plot.pdf', format='pdf')
plt.show()
