import sys
import numpy as np


class Idiosyncratic:
    def __init__(self, class_name, rng, dict_params={}):
        """ """
        class_gen = getattr(sys.modules[__name__], class_name)
        if bool(dict_params):
            self.noise = class_gen(rng=rng,
                                   dict_params=dict_params)
        else:
            self.noise = class_gen(rng=rng)

    def gen(self, num_samples):
        noise = self.noise.gen(num_samples)
        return noise


class Gaussian:
    def __init__(self, rng, dict_params={"std":1.0}):
        self.mean = 0.0
        self.noise_std = dict_params["std"]
        self.rng = rng

    def gen(self, num_samples):
        noise = self.rng.normal(loc=self.mean, scale=self.noise_std,
                                size=num_samples)
        return noise


class Gamma:
    def __init__(self, rng, dict_params={"shape":1, "scale":2.0}):
        # shape: The shape parameter of the gamma distribution (k > 0)
        # scale: The scale parameter of the gamma distribution
        # (theta > 0, default is 1.0)
        self.par_shape = dict_params["shape"]
        self.par_scale = dict_params["scale"]
        self.rng = rng

    def gen(self, num_samples):
        """
        size: The number of samples to generate (optional)
        """
        noise = self.rng.gamma(shape=self.par_shape,
                               scale=self.par_scale,
                               size=num_samples)
        return noise


class Bernoulli:
    def __init__(self, rng, dict_params={"p":0.5}):
        self.p = dict_params["p"]
        self.rng = rng

    def gen(self, num_samples):
        noise = self.rng.binomial(size=num_samples, p=self.p, n=1)
        return noise
