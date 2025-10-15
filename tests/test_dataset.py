from causalspyne.dataset import simpson, visualize_simpson


def test_simpson():
    scenario, treatment, effect = simpson(200)
    visualize_simpson(scenario, treatment, effect)
