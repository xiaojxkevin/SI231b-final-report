# See https://github.com/Enderdead/decoding-lp/blob/master/decode.py for reference
import numpy as np
from scipy import optimize
from scipy import sparse
import math

from multiprocessing import Pool

def decode(A, y):
    """
    e' to be the abs for e \\
    min [0, 1] \dot [f, e'] \\
    s.t. [ -A -I ][ f ] <= [ -y ]
         [  A -I ][ e ] <= [  y ]
    """
    A_ub = np.concatenate((
        np.concatenate((A * -1, A), axis=0),
        np.concatenate((
            -1 * np.identity(A.shape[0]),
            -1 * np.identity(A.shape[0])), axis=0)), axis=1)
    b_ub = np.concatenate((-1 * y, y), axis=0)
    lp_coeff = np.concatenate((np.zeros(A.shape[1]), np.ones(A.shape[0])))

    return optimize.linprog(
        lp_coeff, A_ub=A_ub, b_ub=b_ub,
        bounds=[(None, None) for _ in range(A.shape[1])] + [(0, None) for _ in range(A.shape[0])],
        method='highs')


def form(n:int, m:int, e_p:float):
    A = np.random.normal(size=(m, n), scale=math.sqrt(1 / n))
    x = np.random.normal(size=(n, 1))
    e = sparse.random(m, 1, density=e_p).A
    y = np.matmul(A, x) + e

    return A, x, e, y

def test(n:int, m:int, e_p:float, tol:float):
    A, x, e, y = form(n, m, e_p)
    opt = decode(A, y)
    res = x[:, 0] - opt.x[:n]
    res[np.abs(res) < tol] = 0
    count = np.count_nonzero(res)

    return count, opt.status


def test_mp(args):
    """Procpool compatible wrapper of test"""

    n, m, e_p, tol = args
    return test(n, m, e_p, tol)


def profile(n:int, m:int, error_percent:float, iterations:int, tol:float):
    """Run profile for given parameters
    """

    print(f"Profiling error level: {error_percent}")
    p = Pool(8)
    res = p.map(test_mp, [[n, m, error_percent, tol] for _ in range(iterations)])
    # res = test_mp([n, m, error_percent, tol])

    props = 1. - np.average([r[0] for r in res]) / n
    status = [r[1] for r in res]

    return props, status

def visualize(error_percent:list, success_rates:list, m, n):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(error_percent, success_rates, marker='o')
    plt.title(f'Success Rate of L1 Minimization Recovery with m={m} and n={n}')
    plt.xlabel('Fraction of corrupted entries')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.show()

def main():
    n = 128
    m = 2 * n
    iterations = 100
    tol = 1e-9

    props = []
    e_p = []
    for i in range(10):
        e = 0.05 * (i + 1)
        e_p.append(e)
        props.append(profile(n, m, e, iterations, tol)[0])
    # e_p = 0.05 * (0 + 1)
    # res = profile(n, m, e_p, iterations, tol)
    print(e_p, "\n", props)
    visualize(e_p, props, m, n)

if __name__ == '__main__':
    main()