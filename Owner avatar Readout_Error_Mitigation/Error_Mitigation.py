import tensorcircuit as tc
from tensorcircuit.cloud import apis
from tensorcircuit.noisemodel import NoiseConf
from tensorcircuit.noisemodel import circuit_with_noise
import numpy as np
from scipy.optimize import minimize
from tensorcircuit.cloud import apis
import mthree
apis.set_token("6quf-SXbMHtzHP-bgKYoOEbWGRMFF-TAe4SmExzn4k2zWaaNBFkyuBMLujreoZwdKO4wFWM9EnokXHl-6XIMZHxMODxOt0qNle4QCYerwkuMc358qANAX-A9OUbyDqbV.N4XbhoDOtkhDGbB6UhXHyaa")

n = 13
shots = 10000
online_mode = 0
np.set_printoptions(suppress=True, precision=6)

def get_A(l, r):

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
            res = circ.sample(shots, format = "count_dict_bin", readout_error=readout_error, allow_state=True) 

        for str,c in res.items():
            A[int(str[l : r], 2)][i] += c / shots

    return A

def get_readout_error(n):
    base_error = [0.95, 0.9]
    readout_error = []

    for j in range(n):
        random_change_0 = 0.01 * np.random.rand()
        random_change_1 = 0.01 * np.random.rand()
        readout_error.append([base_error[0] - random_change_0, base_error[1] - random_change_1])

    return readout_error 

readout_error = get_readout_error(n)

A_plus = np.ones((1, 1))
A_plus = np.kron(A_plus, np.linalg.pinv(get_A(0, min(4, n))))
if n > 4 :
    A_plus = np.kron(A_plus, np.linalg.pinv(get_A(4, min(8, n))))
if n > 8 :
    A_plus = np.kron(A_plus, np.linalg.pinv(get_A(8, n)))
# A_plus = np.kron(A_plus, np.linalg.pinv(get_A(0, 9)))
# A_plus = np.kron(A_plus, np.linalg.pinv(get_A(0, 2)))

check_circ = tc.Circuit(n)
check_circ.h(1)
check_circ.h(3)
check_circ.x(4)
check_circ.cz(1,3)
check_circ.cx(3,5)
check_circ.h(8)
print(check_circ.draw())

# res = check_circ.sample(shots, format = "count_dict_bin", allow_state=True)
# real_x = np.zeros(2 ** n)
# for str,c in res.items():
#     real_x[int(str, 2)] = c / shots
# print(real_x)

if online_mode == 1 :
    t = apis.submit_task(provider="tencent", device="tianji_s2", circuit=check_circ, shots=shots)
    res = t.results()
else :
    res = check_circ.sample(shots, format = "count_dict_bin", readout_error=readout_error, allow_state=True)
y = np.zeros(2 ** n)
for str,c in res.items():
    y[int(str, 2)] = c

x = A_plus @ y
print(x)

def vector_to_dict(vector):
    n = len(vector)
    num_bits = len(bin(n - 1)) - 2

    result_dict = {}
    for i in range(n):
        binary_key = format(i, f'0{num_bits}b')
        result_dict[binary_key] = vector[i]

    return result_dict

print(mthree.classes.QuasiDistribution(vector_to_dict(x)).nearest_probability_distribution())
