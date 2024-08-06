from causalspyne.poset_approach import partitions


def test_lattice_enumerate():
    gen = partitions(2)
    # gen = partitions(5)
    # gen = partitions(10)
    # gen = partitions(15)
    len(list(gen))
