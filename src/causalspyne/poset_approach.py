
"""Generate ground truth pair (D, A_S), where D is a DAG and A is the
optimal abstraction of D with respect to score S (e.g. BIC, AIC score).

Given D, construct the poset of graphical abstractions and perform
exhaustive search for the optimal abstraction. This will be a useful
ground truth for evaluating a greedy algorithm that avoids learning D
and directly learns A_S via greedy search from the top of the lattice.

We can start in the observational setting at first and then move to
the interventional setting later.
"""

from typing import Generator, List


def partitions(n: int) -> Generator[List[List[int]], None, None]:
    """Enumerate all partitions of n labelled objects.

    The number of partitions is given by the Bell number, B_n. See
    https://en.wikipedia.org/wiki/Partition_of_a_set#Counting_partitions

    Make sure to memoize this first if we plan to call it many times!
    """
    if n == 1:
        yield [[1]]
    else:
        for partition in partitions(n - 1):
            yield partition + [[n]]
            for idx, part in enumerate(partition):
                prepart = partition[:idx]
                postpart = partition[idx + 1:]
                yield prepart + [part + [n]] + postpart


def abstractions(partial_order) -> Generator[List, None, None]:
    """Enumerate partitions that are consistent with the given partial
    order."""
    yield None
