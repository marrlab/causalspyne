import os
from contextlib import contextmanager


@contextmanager
def chdir(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)
