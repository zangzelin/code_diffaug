from multiprocessing import Pool
import numpy as np


class PoolRunner(object):
    def __init__(self,
        number_point,
        perplexity,
        dist,
        rho,
        gamma,
        v,
        pow=2,
        func_new=None
        ):

        pool = Pool(processes=30)
        result = []
        for dist_row in range(number_point):

            result.append(
                pool.apply_async(sigma_binary_search,
                                 (perplexity, dist[dist_row], rho[dist_row],
                                  gamma, v, pow, func_new)))
        print('start calculate sigma')
        pool.close()
        pool.join()
        sigma_array = []
        for i in result:
            sigma_array.append(i.get())
        self.sigma_array = np.array(sigma_array)
        print("\nMean sigma = " + str(np.mean(sigma_array)))
        print('finish calculate sigma')
        # print(self.sigma_array)
        # input()

    def Getout(self, ):
        return self.sigma_array


def sigma_binary_search(
    fixed_k, dist_row_line, rho_line, gamma, v,
    pow=2, func_new=None):
    """
    Solve equation k_of_sigma(sigma) = fixed_k
    with respect to sigma by the binary search algorithm
    """
    func=func_tdis
    if func_new is not None:
        func=func_new
    
    sigma_lower_limit = 0
    sigma_upper_limit = 1000
    for i in range(50):
        approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
        k_value = func(approx_sigma,
                       dist_row_line,
                       rho_line,
                       gamma,
                       v,
                       pow=pow)
        if k_value < fixed_k:
            sigma_lower_limit = approx_sigma
        else:
            sigma_upper_limit = approx_sigma
        if np.abs(fixed_k - k_value) <= 1e-4:
            break
    return approx_sigma


def func_tdis(sigma, dist_row_line, rho_line, gamma, v, pow=2):
    d = (dist_row_line - rho_line) / sigma
    d[d < 0] = 0
    p = np.power(
        gamma * np.power((1 + d / v), -1 * (v + 1) / 2) * np.sqrt(2 * 3.14),
        pow)
    return np.power(2, np.sum(p))