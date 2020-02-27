import os

import numpy as np

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np

character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
X = xp.array([[0, 1], [1, 0]], dtype=dtype)
Y = xp.array([[0, -1j], [1j, 0]], dtype=dtype)
Z = xp.array([[1, 0], [0, -1]], dtype=dtype)


def _to_tuple(x):
    if np.issubdtype(type(x), np.integer):
        x = (x,)
    return x


def _to_scalar(x):
    if xp != np:
        if isinstance(x, xp.ndarray):
            x = xp.asnumpy(x)
    if isinstance(x, np.ndarray):
        x = x.item(0)
    return x


def make_state(s):
    state = xp.zeros([2] * len(s), dtype=dtype)
    state[tuple([int(i) for i in s])] = 1
    return state


class Qubits:
    """
    Creating qubits.

    Args:
        n_qubits (:class:`int`):
            Number of qubits.

    Attributes:
        state (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The state of qubits.
        n_qubits:
            Number of qubits.
    """

    def __init__(self, n_qubits, n_batch_axes=0):
        self.nq = n_qubits
        self.nb = n_batch_axes
        self.state = xp.zeros([1] * self.nb + [2] * self.nq, dtype=dtype)
        self.state[tuple([0] * (self.nb + self.nq))] = 1

    def reset_state(self):
        self.state = xp.zeros([1] * self.nb + [2] * self.nq, dtype=dtype)
        self.state[tuple([0] * (self.nb + self.nq))] = 1

    def get_state(self, flatten=True):
        """get_state(self, flatten=True)

        Get state.

        Args:
            flatten (:class:`bool`):
                If you set flatten=False, you can get data format used in QuPy.
                otherwise, you get state reformatted to 1D-array.
        """

        if flatten:
            return self.state.reshape(self.state.shape[:self.nb] + (-1,))
        return self.state

    def gate(self, operator, target, control=None, control_0=None, batch_index=()):
        """gate(self, operator, target, control=None, control_0=None)

        Gate method.

        Args:
            operator (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Unitary operator
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operated qubits
            control (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 1
            control_0 (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 0
        """

        target = _to_tuple(target)
        control = _to_tuple(control)
        control_0 = _to_tuple(control_0)
        batch_index = _to_tuple(batch_index)

        c_slice = [slice(None)] * (self.nb + self.nq)
        if control is not None:
            for _c in control:
                c_slice[_c + self.nb] = slice(1, 2)
        if control_0 is not None:
            for _c in control_0:
                c_slice[_c + self.nb] = slice(0, 1)
        c_slice = tuple(c_slice)

        c_idx = list(range(self.nq))
        t_idx = list(range(self.nq))
        for i, _t in enumerate(target):
            t_idx[_t] = self.nq + i
        o_idx = list(range(self.nq, self.nq + len(target))) + list(target)

        o_index = ''.join([character[i + self.nb] for i in o_idx])
        c_index = ''.join([character[i + self.nb] for i in c_idx])
        t_index = ''.join([character[i + self.nb] for i in t_idx])
        b_index = character[: self.nb]
        ob_index = ''.join([character[i] for i in batch_index]) if batch_index is not None else ''
        subscripts = '{},{}->{}'.format(ob_index + o_index, b_index + c_index, b_index + t_index)
        sub_state = xp.einsum(subscripts, operator, self.state[c_slice])
        if self.state.shape[:self.nb] != sub_state.shape[:self.nb]:
            subscripts = '{}...,{}->{}...'.format(b_index, b_index, b_index)
            self.state = xp.einsum(subscripts, self.state, xp.ones(sub_state.shape[: self.nb]))
        self.state[c_slice] = sub_state

    def project(self, target):
        """project(self, target)

        Projection method.

        Args:
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                projected qubits

        Returns:
            :class:`int`: O or 1.
        """

        # state = xp.split(self.state, [1], axis=target + self.nb)
        slice_0 = [slice(None)] * (self.nb + self.nq)
        slice_0[self.nb + target] = slice(0, 1)
        slice_0 = tuple(slice_0)

        state_0 = self.state[slice_0]
        p_0 = xp.sum(
            state_0 * xp.conj(state_0),
            axis=tuple(range(self.nb, self.nb + self.nq)),
            keepdims=True
        ).real
        obs = (xp.random.random(size=p_0.shape) >= p_0)
        multiplier = 1 / xp.sqrt(xp.concatenate(
            (xp.choose(obs, (p_0, np.inf)), xp.choose(obs, (np.inf, 1 - p_0))),
            axis=self.nb + target
        ))
        self.state *= multiplier

        return obs.reshape(obs.shape[:self.nb])

    def expect(self, observable, n_trial=-1):
        """expect(self, observable, n_trial=-1)

        Method to get expected value of observable.

        Args:
            observable (:class:`dict`):
                you have to set
                {'operator1': coef1, 'operator2': coef2, 'operator3': coef3, ...},
                such as {'XIX': 0.32, 'YYZ': 0.11, 'III': 0.02}.
                If you input :class:`dict` as observable,
                this method returns
                :math:`\\sum_i \\mathrm{coef}i \\langle \\psi | \\mathrm{operator}i | \\psi \\rangle`.
            n_trial (:class: int):
                cumulative number.

        Returns:
            :class:`float`: Expected value.
        """

        ret = 0
        org_state = self.state

        for key, value in observable.items():
            self.state = xp.copy(org_state)
            if len(key) != self.nq:
                raise ValueError('Each key length must be {}, but len({}) is {}.'.format(self.nq, key, len(key)))

            for i, op in enumerate(key):
                if op in 'XYZ':
                    self.gate(eval(op), target=i)
                elif op != 'I':
                    raise ValueError('Keys of input must not include {}.'.format(op))

            idx = character[:self.nq]
            e_val = xp.einsum('...{},...{}->...'.format(idx, idx), xp.conj(org_state), self.state).real  # i,i
            if self.nb == 0:
                e_val = _to_scalar(e_val)

            if n_trial > 0:
                probability = (e_val + 1) / 2
                probability = xp.clip(probability, 0, 1)
                e_val = (xp.random.binomial(n_trial, probability) / n_trial) * 2 - 1

            ret += e_val * value

        self.state = org_state
        return ret
