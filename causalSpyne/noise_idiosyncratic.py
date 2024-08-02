import numpy as np


class HyperPars():
    def gen(self):
        return 0.1


class Gaussian():
    def __init__(self, std):
        self.mean = 0
        self.noise_std = std

    def gen(self, num_samples):
        noise = np.random.normal(0, self.noise_std, num_samples)
        return noise


class Idiosyncratic():
    def __init__(self):
        """
        """
        self.noise = Gaussian(HyperPars().gen())

    def gen(self, num_samples):
        noise = self.noise.gen(num_samples)
        return noise
