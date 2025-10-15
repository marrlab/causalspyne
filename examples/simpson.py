from causalspyne.dataset import simpson, visualize
arr = simpson(200)

plot = visualize(arr)
plot.show()
plot.savefig('simpson_scatter_plot.pdf', format='pdf')
plot.close()
