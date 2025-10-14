from causalspyne.dataset import simpson, visualize


def test_simpson():
    arr = simpson(200)
    plot = visualize(arr)
    plot.close()
