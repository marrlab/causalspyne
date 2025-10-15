from causalspyne.dataset import simpson, visualize


def test_simpson():
    scenario, treatment, effect = simpson(200)
    visualize(scenario, treatment, effect)
