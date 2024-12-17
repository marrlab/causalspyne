import sys
import numpy as np


class HyperPars:
    def gen(self):
        return 0.1


class Gaussian:
    def __init__(self, rng, params):
        self.mean = 0
        if "gaussian_std" in params:
            self.noise_std = params["gaussian_std"]
        else:
            self.noise_std = HyperPars().gen()

        self.rng = rng

    def gen(self, num_samples):
        noise = self.rng.normal(0, self.noise_std, num_samples)
        return noise


class Gamma:
    def __init__(self, rng, params):
        # shape: The shape parameter of the gamma distribution (k > 0)
        # scale: The scale parameter of the gamma distribution
        # (theta > 0, default is 1.0)
        if "gamma_shape" in params:
            self.par_shape = params["gamma_shape"]
        else:
            self.par_shape = 1
        if "gamma_scale" in params:
            self.par_scale = params["gamma_scale"]
        else:
            self.par_scale = 2.0
        self.rng = rng

    def gen(self, num_samples):
        """
        size: The number of samples to generate (optional)
        """
        noise = self.rng.gamma(shape=self.par_shape,
                               scale=self.par_scale,
                               size=num_samples)
        return noise


class Idiosyncratic:
    def __init__(self, rng, class_name="Gamma", **additional_info):
        """ """
        class_gen = getattr(sys.modules[__name__], class_name)
        self.noise = class_gen(rng, additional_info)

    def gen(self, num_samples):
        noise = self.noise.gen(num_samples)
        return noise
