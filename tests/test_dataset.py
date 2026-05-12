from causalspyne.dataset import simpson, visualize_simpson


def test_simpson(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scenario, treatment, effect = simpson(200)
    visualize_simpson(scenario, treatment, effect)
