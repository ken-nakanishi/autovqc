#########################################################
# Calculate the ground energy of LiH with bond distance #
#########################################################

import os
import toml

import numpy as np

from autovqc import CircuitOptimizer

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np

I = np.array([[1, 0], [0, 1]], dtype=dtype)
X = np.array([[0, 1], [1, 0]], dtype=dtype)
Y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
Z = np.array([[1, 0], [0, -1]], dtype=dtype)


def _kron(*args):
    args = list(args)
    while len(args) != 1:
        x = args.pop()
        args[-1] = np.kron(args[-1], x)
    return args[0]


def solve(ham):
    h_mat = None
    for k, v in ham.items():
        if h_mat is None:
            h_mat = np.zeros((2 ** len(k), 2 ** len(k)), dtype=np.complex128)
        tmp = [eval(op) for op in k]
        h_mat += _kron(*tmp) * v
    _evals, _evecs = np.linalg.eigh(h_mat)
    eig_args = np.argsort(_evals)
    eig_vals = _evals[eig_args]
    eig_vecs = _evecs.T[eig_args]
    return eig_vals, eig_vecs


def main():
    with open(os.path.join('data', 'LiH.toml'), 'r') as f:
        hamiltonian = toml.load(f)

    connections = [(0, 1), (1, 2), (2, 3)]

    circ_opt = CircuitOptimizer(
        hamiltonian,
        connections,
        n_depth=3,
        batchsize=10,
        maxiter=256
    )

    for i in range(5):
        if i != 0:
            circ_opt.update()
        result = circ_opt.get_result()
        print(result['loss'])
        print(result['targets_list'])


if __name__ == '__main__':
    with open(os.path.join('data', 'LiH.toml'), 'r') as f:
        hamiltonian = toml.load(f)

    eig_vals, _ = solve(hamiltonian)
    print(eig_vals[0])

    main()
