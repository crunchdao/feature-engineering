"""
Resources: https://github.com/ninfueng/lloyd-max-quantizer
           https://en.wikipedia.org/wiki/Quantization_(signal_processing)#Neglecting_the_entropy_constraint:_Lloyd%E2%80%
           https://web.stanford.edu/class/ee398a/handouts/lectures/05-Quantization.pdf

"""

"""Collection of utility functions for Lloyd Max Quantizer.
@author: Ninnart Fuengfusin
"""
import numpy as np
import scipy.integrate as integrate


def normal_dist(x, mean=0.0, vari=1.0):
    """A normal distribution function created to use with scipy.integral.quad"""
    return (1.0 / (np.sqrt(2.0 * np.pi * vari))) * np.exp(
        (-np.power((x - mean), 2.0)) / (2.0 * vari)
    )


def expected_normal_dist(x, mean=0.0, vari=1.0):
    """A expected value of normal distribution function which created to use with scipy.integral.quad"""
    return (x / (np.sqrt(2.0 * np.pi * vari))) * np.exp(
        (-np.power((x - mean), 2.0)) / (2.0 * vari)
    )


def MSE_loss(x, x_hat_q):
    """Find the mean square loss between x (orginal signal) and x_hat (quantized signal)
    Args:
        x: the signal without quantization
        x_hat_q: the signal of x after quantization
    Return:
        MSE: mean square loss between x and x_hat_q
    """
    # protech in case of input as tuple and list for using with numpy operation
    x = np.array(x)
    x_hat_q = np.array(x_hat_q)
    assert np.size(x) == np.size(x_hat_q)
    MSE = np.sum(np.power(x - x_hat_q, 2)) / np.size(x)
    return MSE


class LloydMaxQuantizer:
    """A class for iterative Lloyd Max quantizer.
    This quantizer is created to minimize amount SNR between the orginal signal
    and quantized signal.
    """

    @staticmethod
    def start_repre(x, bit):
        """
        Generate representations of each threshold using
        Args:
            x: input signal for
            bit: amount of bit
        Return:
            threshold:
        """
        assert isinstance(bit, int)
        x = np.array(x)
        num_repre = bit
        step = (np.max(x) - np.min(x)) / num_repre

        middle_point = np.mean(x)
        repre = np.array([])
        # his code works for even bins, because he assumes power of two, we have 5. Let's modify the code.
        repre = np.append(repre, middle_point)
        for i in range(int(num_repre / 2)):
            repre = np.append(repre, middle_point + (i + 1) * step)
            repre = np.insert(repre, 0, middle_point - (i + 1) * step)
        return repre

    @staticmethod
    def threshold(repre):
        """ """
        t_q = np.zeros(np.size(repre) - 1)
        for i in range(len(repre) - 1):
            t_q[i] = 0.5 * (repre[i] + repre[i + 1])
        return t_q

    @staticmethod
    def represent(thre, expected_dist, dist):
        """ """
        thre = np.array(thre)
        x_hat_q = np.zeros(np.size(thre) + 1)
        # prepare for all possible integration range
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)

        for i in range(len(thre) - 1):
            x_hat_q[i] = integrate.quad(expected_dist, thre[i], thre[i + 1])[0] / (
                integrate.quad(dist, thre[i], thre[i + 1])[0]
            )
        return x_hat_q

    @staticmethod
    def quant(x, thre, repre):
        """Quantization operation."""
        thre = np.append(thre, np.inf)
        thre = np.insert(thre, 0, -np.inf)
        x_hat_q = np.zeros(np.shape(x))
        for i in range(len(thre) - 1):
            if i == 0:
                x_hat_q = np.where(
                    np.logical_and(x > thre[i], x <= thre[i + 1]),
                    np.full(np.size(x_hat_q), repre[i]),
                    x_hat_q,
                )
            elif i == range(len(thre))[-1] - 1:
                x_hat_q = np.where(
                    np.logical_and(x > thre[i], x <= thre[i + 1]),
                    np.full(np.size(x_hat_q), repre[i]),
                    x_hat_q,
                )
            else:
                x_hat_q = np.where(
                    np.logical_and(x > thre[i], x < thre[i + 1]),
                    np.full(np.size(x_hat_q), repre[i]),
                    x_hat_q,
                )
        return x_hat_q


def quantize(x):

    bits = 7
    iterations = 10

    repre = LloydMaxQuantizer.start_repre(x, bits)
    min_loss = 1.0

    for i in range(iterations):
        thre = LloydMaxQuantizer.threshold(repre)
        repre = LloydMaxQuantizer.represent(thre, expected_normal_dist, normal_dist)
        x_hat_q = LloydMaxQuantizer.quant(x, thre, repre)
        loss = MSE_loss(x, x_hat_q)

        # Keep the threhold and representation that has the lowest MSE loss.
        if min_loss > loss:
            min_loss = loss
            min_thre = thre
            min_repre = repre

    # x_hat_q with the lowest amount of loss.
    best_x_hat_q = LloydMaxQuantizer.quant(x, min_thre, min_repre)

    unique = np.unique(best_x_hat_q)
    discrete = np.linspace(0, 1, bits)
    for i in range(len(unique))[::-1]:
        best_x_hat_q[best_x_hat_q == unique[i]] = discrete[i]

    return best_x_hat_q


def hard_quantize(x, bins):
    quant_index_list = []
    limit_old = -np.inf
    for idx, el in enumerate(bins):
        limit = x.quantile(el)
        quant_index = (x <= limit) & (x >= limit_old)
        quant_index_list.append(quant_index)
        limit_old = limit

    for idx, qi in enumerate(quant_index_list):
        x.loc[qi] = round(idx / (len(bins) - 1), 2)
    return x
