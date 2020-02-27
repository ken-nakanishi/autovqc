import os
from copy import deepcopy

import numpy as np

from autovqc.qubit import Qubits
from autovqc.operator import Z, batch_rx, batch_rz

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np


class Circuit:
    def __init__(self, n_qubits, targets_list):
        self.targets_list = targets_list
        self.n_qubits = n_qubits
        self.n_params = self.n_qubits * 2 + len(targets_list) * 4

    def __call__(self, qubits, params):
        p = iter(params)
        for i in range(self.n_qubits):
            qubits.gate(batch_rx(next(p)), target=i, batch_index=(0, 1))
            qubits.gate(batch_rz(next(p)), target=i, batch_index=(0, 1))
        for t in self.targets_list:
            qubits.gate(Z, target=t[0], control=t[1])
            qubits.gate(batch_rx(next(p)), target=t[0], batch_index=(0, 1))
            qubits.gate(batch_rx(next(p)), target=t[1], batch_index=(0, 1))
            qubits.gate(batch_rz(next(p)), target=t[0], batch_index=(0, 1))
            qubits.gate(batch_rz(next(p)), target=t[1], batch_index=(0, 1))


class Model:
    def __init__(self, hamiltonian, targets_list, n_shots=-1):
        self.hamiltonian = hamiltonian
        self.targets_list = targets_list
        self.n_qubits = len(next(iter(self.hamiltonian)))
        self.n_shots = n_shots
        self.qubits = Qubits(self.n_qubits, n_batch_axes=2)
        self.circ = Circuit(self.n_qubits, targets_list)
        self.n_params = self.circ.n_params

    def __call__(self, params):
        self.qubits.reset_state()
        self.circ(self.qubits, params)
        avg_ham = self.qubits.expect(self.hamiltonian, self.n_shots)
        return avg_ham


def param_optimize(fun, x, maxiter=512, reset_interval=32, eps=1e-32, callback=None):

    n_params = len(x)  # x: list of xp.array(shape: (1, n_batch))
    recycle_z0 = None
    funcalls = 0

    for niter in range(maxiter):

        idx = niter % n_params

        if reset_interval > 0:
            if niter % reset_interval == 0:
                recycle_z0 = None

        p = deepcopy(x)
        if recycle_z0 is None:
            p[idx] = np.pi / 2 * xp.array([[0], [1], [-1]]) + x[idx]
            z0, z1, z3 = xp.split(fun(p), 3, axis=0)
            funcalls += 3
        else:
            z0 = recycle_z0
            p[idx] = np.pi / 2 * xp.array([[1], [-1]]) + x[idx]
            z1, z3 = xp.split(fun(p), 2, axis=0)
            funcalls += 2

        z2 = z1 + z3 - z0
        c = (z1 + z3) / 2
        a = xp.sqrt((z0 - z2) ** 2 + (z1 - z3) ** 2) / 2
        b = xp.arctan((z1 - z3) / ((z0 - z2) + eps * (z0 == z2))) + x[idx]
        b += 0.5 * np.pi * (1 + xp.sign((z0 - z2) + eps * (z0 == z2)))

        x[idx] = b
        recycle_z0 = c - a

        if callback is not None:
            callback(deepcopy(x))

    return {'x': x, 'loss': fun(deepcopy(x)), 'nfev': funcalls}


class CircuitOptimizer:

    def __init__(self, hamiltonian, connections, n_entangular=-1, n_depth=-1, batchsize=100, maxiter=512, n_shots=-1):
        if n_entangular * n_depth > 0:
            raise ValueError('You should set n_entangular xor n_depth.')
        self.hamiltonian = hamiltonian
        self.connections = connections
        self.n_entangular = n_entangular
        self.n_depth = n_depth
        self.batchsize = batchsize
        self.maxiter = maxiter
        self.n_shots = n_shots
        self.best_result = {
            'loss': np.inf,
            'params': None,
            'targets_list': None
        }
        self.update()

    def call(self, targets_list, params=None):
        model = Model(self.hamiltonian, targets_list, self.n_shots)
        if params is None:
            params = xp.random.random(size=(model.n_params, self.batchsize)) * 2 * np.pi
        res = param_optimize(model, xp.split(params, model.n_params), maxiter=self.maxiter)
        argmin = xp.argmin(res['loss'])  # res['loss'].shape=(1, n_batch)
        min_loss = res['loss'][0, argmin]
        opt_params = xp.concatenate(res['x'])[:, argmin]
        return min_loss, opt_params

    def update(self):
        if self.n_depth >= 0:
            self._update_n_depth()
        else:
            self._update_n_entangular()

    def _update_n_entangular(self):
        tl = [self.connections[i] for i in np.random.randint(len(self.connections), size=self.n_entangular)]
        loss, params = self.call(tl)
        if loss < self.best_result['loss']:
            self.best_result = {
                'loss': loss,
                'params': params,
                'targets_list': tl
            }

    def _update_n_depth(self):
        tl = []  # target_list
        for _ in range(self.n_depth):
            rest = self.connections
            tl_one_depth = []
            while len(rest) != 0:
                idx = np.random.randint(len(rest))
                pickee = rest[idx]
                tl_one_depth.append(pickee)
                rest = list(filter(lambda x: ((x[0] not in pickee) and (x[1] not in pickee)), rest))

            tl += sorted(tl_one_depth, key=lambda x: x[0])

        loss, params = self.call(tl)
        if loss < self.best_result['loss']:
            self.best_result = {
                'loss': loss,
                'params': params,
                'targets_list': tl
            }

    def get_result(self):
        return self.best_result

    def calc(self, params, targets_list):
        model = Model(self.hamiltonian, targets_list, self.n_shots)
        loss = model([[[p]] for p in params])
        return loss[0][0]
