import numpy as np


class HyperPars:
    def gen(self):
        return 0.1


class Gaussian:
    def __init__(self, std, rng):
        self.mean = 0
        self.noise_std = std
        self.rng = rng

    def gen(self, num_samples):
        noise = self.rng.normal(0, self.noise_std, num_samples)
        return noise


class Idiosyncratic:
    def __init__(self, rng):
        """ """
        self.noise = Gaussian(HyperPars().gen(), rng)

    def gen(self, num_samples):
        noise = self.noise.gen(num_samples)
        return noise
