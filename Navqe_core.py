# NA-VQA implementation

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    HAVE_AER_EST = True
except Exception:
    from qiskit.primitives import Estimator as AerEstimator
    HAVE_AER_EST = False

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

def dagger_circuit(circ: QuantumCircuit) -> QuantumCircuit:
  
    circ_no_meas = QuantumCircuit(circ.num_qubits)
    for instruction in circ.data:
        if instruction.operation.name != 'measure':
            circ_no_meas.append(instruction.operation, instruction.qubits, instruction.clbits)
    
   
    inv = circ_no_meas.inverse()
    inv.name = f"{circ.name}^\u2020"
    return inv

def fold_circuit(circ: QuantumCircuit, scale: float) -> QuantumCircuit:
    if scale < 1.0:
        raise ValueError("scale must be >= 1")
    if np.isclose(scale, 1.0):
        return circ
    k = int(np.round(scale))
    if k % 2 == 0:
        k += 1
    if k == 1:
        return circ
    front = circ
    pair = QuantumCircuit(circ.num_qubits)
    pair.compose(dagger_circuit(circ), inplace=True)
    pair.compose(circ, inplace=True)
    times = (k - 1) // 2
    folded = QuantumCircuit(circ.num_qubits)
    folded.compose(front, inplace=True)
    for _ in range(times):
        folded.compose(pair, inplace=True)
    folded.name = f"{circ.name}_fold{k}"
    return folded

def expectation_from_counts_z(counts: Dict[str, int], qubits: List[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for bitstr, c in counts.items():
        val = 1
        for q in qubits:
            bit = bitstr[::-1][q]
            val *= 1 if bit == '0' else -1
        exp += val * c
    return exp / total

@dataclass
class NoiseEstimates:
    depolarizing_p: float
    amplitude_damping_gamma: float

class NoiseCalibrator:
    def __init__(self, backend, shots: int = 2000, seed: Optional[int] = 1234):
        self.backend = backend
        self.shots = shots
        self.seed = seed

    def _run_counts(self, circ: QuantumCircuit) -> Dict[str, int]:
        tr = transpile(circ, backend=self.backend, seed_transpiler=self.seed, optimization_level=1)
        job = self.backend.run(tr, shots=self.shots, seed_simulator=self.seed)
        return job.result().get_counts()

    def estimate_depolarizing(self, num_qubits: int = 1, scales: List[float] = [1, 3, 5]) -> float:
        U = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            U.h(q)
        for q in range(num_qubits - 1):
            U.cx(q, q + 1)
        for q in range(num_qubits):
            U.rx(0.4, q)
        U_bar = dagger_circuit(U)
        base = QuantumCircuit(num_qubits)
        base.compose(U, inplace=True)
        base.compose(U_bar, inplace=True)
        base.measure_all()
        Es = []
        for s in scales:
            folded = fold_circuit(base, s)
            counts = self._run_counts(folded)
            E = expectation_from_counts_z(counts, list(range(num_qubits)))
            Es.append((s, max(min(E, 0.999999), -0.999999)))
        x = np.array([s for s, _ in Es], dtype=float)
        y = np.array([np.log(max(min((E + 1) / 2.0, 0.999999), 1e-6)) for _, E in Es], dtype=float)
        k = -np.polyfit(x, y, 1)[0]
        p_eff = 1.0 - np.exp(-k)
        return float(np.clip(p_eff, 0.0, 0.5))

    def estimate_amplitude_damping(self, num_qubits: int = 1, scales: List[float] = [1, 3, 5]) -> float:
        U = QuantumCircuit(num_qubits)
        U.x(0)
        U.rx(0.3, 0)
        U_bar = dagger_circuit(U)
        base = QuantumCircuit(num_qubits)
        base.compose(U, inplace=True)
        base.compose(U_bar, inplace=True)
        base.measure_all()
        P1s = []
        for s in scales:
            folded = fold_circuit(base, s)
            counts = self._run_counts(folded)
            total = sum(counts.values())
            p1 = 0.0
            for bitstr, c in counts.items():
                if bitstr[::-1][0] == '1':
                    p1 += c
            p1 = p1 / max(total, 1)
            P1s.append((s, max(min(p1, 0.999999), 1e-6)))
        x = np.array([s for s, _ in P1s], dtype=float)
        y = np.array([p for _, p in P1s], dtype=float)
        lam = -np.polyfit(x, np.log(y), 1)[0]
        gamma_eff = 1.0 - np.exp(-lam)
        return float(np.clip(gamma_eff, 0.0, 1.0))

    def estimate_noise(self, num_qubits: int = 2) -> NoiseEstimates:
        p = self.estimate_depolarizing(num_qubits=num_qubits, scales=[1, 3, 5])
        g = self.estimate_amplitude_damping(num_qubits=num_qubits, scales=[1, 3, 5])
        return NoiseEstimates(p, g)

def layered_hardware_efficient_ansatz(n: int, depth: int, params: np.ndarray) -> QuantumCircuit:
    circ = QuantumCircuit(n)
    idx = 0
    for _ in range(depth):
        for q in range(n):
            circ.ry(params[idx], q); idx += 1
        for q in range(n - 1):
            circ.cx(q, q + 1)
        for q in range(n):
            circ.rz(params[idx], q); idx += 1
    return circ

def transverse_field_ising(n: int, h: float = 1.0) -> SparsePauliOp:
    terms, coeffs = [], []
    for i in range(n - 1):
        p = ['I'] * n
        p[i], p[i + 1] = 'Z', 'Z'
        terms.append(''.join(p[::-1])); coeffs.append(1.0)
    for i in range(n):
        p = ['I'] * n
        p[i] = 'X'
        terms.append(''.join(p[::-1])); coeffs.append(h)
    return SparsePauliOp.from_list(list(zip(terms, coeffs)))

class SPSA:
    def __init__(self, a: float = 0.2, c: float = 0.2, alpha: float = 0.602, gamma: float = 0.101, maxiter: int = 50, seed: int = 1234):
        self.a0, self.c0, self.alpha, self.gamma, self.maxiter = a, c, alpha, gamma, maxiter
        self.rng = np.random.default_rng(seed)
    def step_sizes(self, k: int) -> Tuple[float, float]:
        return self.a0 / ((k + 1) ** self.alpha), self.c0 / ((k + 1) ** self.gamma)
    def minimize(self, f, x0: np.ndarray, callback=None) -> Tuple[np.ndarray, List[float]]:
        x = x0.copy(); hist = []
        for k in range(self.maxiter):
            a_k, c_k = self.step_sizes(k)
            delta = self.rng.choice([-1.0, 1.0], size=x.shape)
            f_plus = f(x + c_k * delta)
            f_minus = f(x - c_k * delta)
            gk = (f_plus - f_minus) / (2.0 * c_k * delta)
            x = x - a_k * gk
            val = f(x); hist.append(val)
            if callback is not None: callback(k, x, val)
        return x, hist

@dataclass
class AdaptConfig:
    min_depth: int = 1
    max_depth: int = 8
    base_depth: int = 2
    base_step_a: float = 0.25
    base_step_c: float = 0.25
    noise_depth_slope: float = 10.0
    min_step_scale: float = 0.2
    max_step_scale: float = 1.5

class NAVQE:
    def __init__(self, H: SparsePauliOp, n_qubits: int, backend=None, shots: int = 2000, seed: int = 1234, adapt: AdaptConfig = AdaptConfig(), noise_recalib_every: int = 3):
        self.H = H
        self.n = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.seed = seed
        self.adapt = adapt
        self.noise_recalib_every = noise_recalib_every
        self.calib = NoiseCalibrator(self.backend, shots=shots, seed=seed)
        self.current_noise = NoiseEstimates(0.05, 0.05)
        self.depth = adapt.base_depth
        self.spsa = SPSA(a=adapt.base_step_a, c=adapt.base_step_c, maxiter=30, seed=seed)
        
        self.estimator = AerEstimator()

    def _update_adaptation(self):
        sev = max(self.current_noise.depolarizing_p, self.current_noise.amplitude_damping_gamma)
        target_depth = int(round(np.clip(self.adapt.base_depth + self.adapt.noise_depth_slope * (0.10 - sev), self.adapt.min_depth, self.adapt.max_depth)))
        self.depth = target_depth
        step_scale = float(np.clip(1.0 - 2.0 * sev, self.adapt.min_step_scale, self.adapt.max_step_scale))
        self.spsa = SPSA(a=self.adapt.base_step_a * step_scale, c=self.adapt.base_step_c * step_scale, maxiter=self.spsa.maxiter, seed=self.seed)

    def _param_count(self, depth: int) -> int:
        return depth * (2 * self.n)

    def energy(self, params: np.ndarray) -> float:
        circ = layered_hardware_efficient_ansatz(self.n, self.depth, params)
       
        job = self.estimator.run([circ], [self.H], shots=self.shots, seed=self.seed)
        return float(job.result().values[0])

    def run(self, init_params: Optional[np.ndarray] = None, max_outer_iters: int = 4) -> Tuple[float, np.ndarray, List[float]]:
        self.current_noise = self.calib.estimate_noise(num_qubits=self.n)
        self._update_adaptation()
        pcount = self._param_count(self.depth)
        if init_params is None or init_params.shape[0] != pcount:
            rng = np.random.default_rng(self.seed)
            init_params = rng.uniform(low=-np.pi, high=np.pi, size=pcount)
        best_val, best_params, history = math.inf, init_params.copy(), []
        x = init_params.copy()
        for outer in range(max_outer_iters):
            if outer == 0 or (outer % self.noise_recalib_every == 0):
                self.current_noise = self.calib.estimate_noise(num_qubits=self.n)
                self._update_adaptation()
                pcount = self._param_count(self.depth)
                if x.shape[0] != pcount:
                    x = np.resize(x, pcount)
            def f_eval(theta): return self.energy(theta)
            def cb(k, th, val): history.append(val)
            x, _ = self.spsa.minimize(f_eval, x, callback=cb)
            val = f_eval(x)
            if val < best_val:
                best_val, best_params = val, x.copy()
        return best_val, best_params, history
