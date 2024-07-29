import tensorcircuit as tc
from tensorcircuit.cloud import apis
from tensorcircuit.noisemodel import NoiseConf
from tensorcircuit.noisemodel import circuit_with_noise
import numpy as np
from scipy.optimize import minimize
from tensorcircuit.cloud import apis
import mthree

def get_readout_error(n):
    base_error = [0.95, 0.9]
    readout_error = []

    for j in range(n):
        random_change_0 = 0.01 * np.random.rand()
        random_change_1 = 0.01 * np.random.rand()
        readout_error.append([base_error[0] - random_change_0, base_error[1] - random_change_1])

    return readout_error 

def get_A(n, l, r, online_mode):

    shots = 10000

    A = np.zeros((2 ** (r - l), 2 ** (r - l)))

    for i in range(0, (2 ** (r - l))) :

        circ = tc.Circuit(n)

        for j in range(0, r - l):
            if ((i & (2 ** j)) != 0) :
                circ.x(r - 1 - j)

        if online_mode == 1:
            t = apis.submit_task(provider="tencent", device="tianji_s2", circuit=circ, shots=shots)
            res = t.results()
        else :
            res = circ.sample(shots, format = "count_dict_bin", readout_error=get_readout_error(n), allow_state=True) 

        for str,c in res.items():
            A[int(str[l : r], 2)][i] += c / shots

    return A

def get_Aplus(n, online_mode):
    A_plus = np.ones((1, 1))
    A_plus = np.kron(A_plus, np.linalg.pinv(get_A(n, 0, min(4, n),online_mode)))
    if n > 4 :
        A_plus = np.kron(A_plus, np.linalg.pinv(get_A(n, 4, min(8, n),online_mode)))
    if n > 8 :
        A_plus = np.kron(A_plus, np.linalg.pinv(get_A(n, 8, n),online_mode))
    return A_plus

def vector_to_dict(vector):
    n = len(vector)
    num_bits = len(bin(n - 1)) - 2

    result_dict = {}
    for i in range(n):
        binary_key = format(i, f'0{num_bits}b')
        result_dict[binary_key] = vector[i]

    return result_dict

def get_res(n, res, A_plus):
    y = np.zeros(2 ** n)
    for str,c in res.items():
        y[int(str, 2)] = c

    return mthree.classes.QuasiDistribution(vector_to_dict(A_plus @ y)).nearest_probability_distribution()
