from poset_approach import partitions


def test_lattice_enumerate():
    gen = partitions(2)
    gen = partitions(5)
    list(gen)
