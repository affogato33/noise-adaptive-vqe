# Enhanced NA-VQE Implementation
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time


from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, ReadoutError


# SPSA Optimizer

class SimpleSPSA:
   
    
    def __init__(self, maxiter: int = 100, a: float = 0.1, c: float = 0.1, seed: int = 1234):
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.rng = np.random.default_rng(seed)
    
    def minimize(self, fun, x0):
        
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_fun = fun(x)
        
        for k in range(self.maxiter):
            # Step sizes
            a_k = self.a / ((k + 1) ** 0.602)
            c_k = self.c / ((k + 1) ** 0.101)
            
            # Random perturbation
            delta = self.rng.choice([-1.0, 1.0], size=x.shape)
            
            # Function evaluations
            f_plus = fun(x + c_k * delta)
            f_minus = fun(x - c_k * delta)
            
            # Gradient estimate
            gk = (f_plus - f_minus) / (2.0 * c_k * delta)
            
            # Update parameters
            x = x - a_k * gk
            
            # Track best
            current_fun = fun(x)
            if current_fun < best_fun:
                best_fun = current_fun
                best_x = x.copy()
        
        
        class Result:
            def __init__(self, x, fun):
                self.x = x
                self.fun = fun
        
        return Result(best_x, best_fun)



@dataclass
class NoiseEstimates:
    depolarizing_p: float
    amplitude_damping_gamma: float

@dataclass
class AdaptiveConfig:
    min_depth: int = 1
    max_depth: int = 4
    base_depth: int = 2
    recalibration_frequency: int = 3

@dataclass
class NoiseConfig:
    depolarizing_p: float = 0.01
    amplitude_damping_gamma: float = 0.01
    readout_error: float = 0.01
    noise_type: str = "depolarizing"  


#  Noise Injection

class NoiseInjector:
    """Inject realistic noise into quantum circuits with multiple noise models"""
    
    def __init__(self, noise_config: NoiseConfig = NoiseConfig()):
        self.config = noise_config
        self._build_noise_model()
    
    def _build_noise_model(self):
        """Build Qiskit noise model with selected noise type"""
        self.noise_model = NoiseModel()
        
        if self.config.noise_type == "depolarizing" or self.config.noise_type == "combined":
            if self.config.depolarizing_p > 0:
                # 1-qubit depolarizing error
                error_1q = depolarizing_error(self.config.depolarizing_p, 1)
                self.noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz'])
                
                # 2-qubit depolarizing error
                error_2q = depolarizing_error(self.config.depolarizing_p, 2)
                self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        if self.config.noise_type == "amplitude_damping" or self.config.noise_type == "combined":
            if self.config.amplitude_damping_gamma > 0:
                # 1-qubit amplitude damping error
                error_1q = amplitude_damping_error(self.config.amplitude_damping_gamma)
                self.noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz'])
                
                # 2-qubit amplitude damping error
                error_2q = amplitude_damping_error(self.config.amplitude_damping_gamma)
                self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        if self.config.noise_type == "readout" or self.config.noise_type == "combined":
            if self.config.readout_error > 0:
                # Readout error
                readout_error = ReadoutError([[1 - self.config.readout_error, self.config.readout_error],
                                            [self.config.readout_error, 1 - self.config.readout_error]])
                self.noise_model.add_all_qubit_readout_error(readout_error)
    
    def get_noisy_backend(self, shots: int = 4000) -> AerSimulator:
        """Get backend with injected noise"""
        backend = AerSimulator(noise_model=self.noise_model, shots=shots, seed_simulator=1234)
        return backend


# Noise Calibration


class NoiseCalibrator:
   
    
    def __init__(self, backend, shots: int = 2000, seed: Optional[int] = 1234):
        self.backend = backend
        self.shots = shots
        self.seed = seed
    
    def _run_counts(self, circ: QuantumCircuit) -> Dict[str, int]:
        tr = transpile(circ, backend=self.backend, seed_transpiler=self.seed, optimization_level=1)
        job = self.backend.run(tr, shots=self.shots)
        return job.result().get_counts()
    
    def estimate_depolarizing(self, num_qubits: int = 2) -> float:
       
        #  calibration circuit
        circ = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            circ.h(q)
        circ.measure_all()
        
        counts = self._run_counts(circ)
        total = sum(counts.values())
        
        # Estimation of noise from deviation from ideal
        ideal_prob = 1.0 / (2 ** num_qubits)
        actual_prob = counts.get('0' * num_qubits, 0) / total if total > 0 else 0
        
        #  noise estimation (2)
        noise_level = abs(actual_prob - ideal_prob) * 2
        return min(noise_level, 0.5)
    
    def estimate_amplitude_damping(self, num_qubits: int = 2) -> float:
      
        #  calibration circuit
        circ = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            circ.x(q)
        circ.measure_all()
        
        counts = self._run_counts(circ)
        total = sum(counts.values())
        
        # Estimation from |1⟩ state probability
        prob_1 = counts.get('1' * num_qubits, 0) / total if total > 0 else 0
        ideal_prob = 1.0
        
        noise_level = abs(prob_1 - ideal_prob)
        return min(noise_level, 1.0)
    
    def estimate_readout_error(self, num_qubits: int = 2) -> float:
       
        
        circ = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            circ.x(q)
        circ.measure_all()
        
        counts = self._run_counts(circ)
        total = sum(counts.values())
        
       
        prob_1 = counts.get('1' * num_qubits, 0) / total if total > 0 else 0
        ideal_prob = 1.0
        
        readout_error = abs(prob_1 - ideal_prob)
        return min(readout_error, 0.5)
    
    def estimate_noise(self, num_qubits: int = 2) -> NoiseEstimates:
       
        p = self.estimate_depolarizing(num_qubits=num_qubits)
        g = self.estimate_amplitude_damping(num_qubits=num_qubits)
        return NoiseEstimates(depolarizing_p=p, amplitude_damping_gamma=g)


#Ansatz and Hamiltonians

def hardware_efficient_ansatz(n: int, depth: int) -> QuantumCircuit:
    
    circ = QuantumCircuit(n)
    params = []
    
    for d in range(depth):
        # RY rotations
        for q in range(n):
            param = Parameter(f'θ_{d}_{q}')
            circ.ry(param, q)
            params.append(param)
        
        # Entangling gates
        for q in range(n - 1):
            circ.cx(q, q + 1)
        
        # RZ rotations
        for q in range(n):
            param = Parameter(f'φ_{d}_{q}')
            circ.rz(param, q)
            params.append(param)
    
    return circ, params

def transverse_field_ising(n: int, h: float = 1.0) -> SparsePauliOp:
  
    terms, coeffs = [], []
    
    # ZZ interactions
    for i in range(n - 1):
        pauli = ['I'] * n
        pauli[i] = 'Z'
        pauli[i + 1] = 'Z'
        terms.append(''.join(pauli[::-1]))
        coeffs.append(1.0)
    
    # X fields
    for i in range(n):
        pauli = ['I'] * n
        pauli[i] = 'X'
        terms.append(''.join(pauli[::-1]))
        coeffs.append(h)
    
    return SparsePauliOp.from_list(list(zip(terms, coeffs)))



class StandardVQE:
  
    
    def __init__(self, H: SparsePauliOp, n_qubits: int, backend=None, 
                 shots: int = 2000, depth: int = 2):
        self.H = H
        self.n = n_qubits
        self.backend = backend or AerSimulator(shots=shots, seed_simulator=1234)
        self.shots = shots
        self.depth = depth
        
        #  ansatz
        self.ansatz, self.params = hardware_efficient_ansatz(n_qubits, depth)
        
      
        self.estimator = AerEstimator()
    
    def energy(self, param_values: List[float]) -> float:
        
   
        param_dict = {param: val for param, val in zip(self.params, param_values)}
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        
     
        job = self.estimator.run([bound_circuit], [self.H])
        result = job.result()
        
        
        try:
            return float(result.values[0])
        except AttributeError:
            try:
                return float(result.data.evs[0])
            except AttributeError:
                return float(result[0])
    
    def run(self, max_iterations: int = 50) -> Tuple[float, List[float], List[float]]:
       
        start_time = time.time()
        
    
        init_params = np.random.uniform(-np.pi, np.pi, len(self.params))
        
     
        optimizer = SimpleSPSA(maxiter=max_iterations)
        
        def cost_function(params):
            return self.energy(params)
        
        result = optimizer.minimize(cost_function, init_params)
        
        runtime = time.time() - start_time
        
        return result.fun, result.x.tolist(), [result.fun]

class NAVQE:
    """Noise-Adaptive VQE"""
    
    def __init__(self, H: SparsePauliOp, n_qubits: int, backend=None, 
                 shots: int = 2000, config: AdaptiveConfig = AdaptiveConfig()):
        self.H = H
        self.n = n_qubits
        self.backend = backend or AerSimulator(shots=shots, seed_simulator=1234)
        self.shots = shots
        self.config = config
        
    
        self.calib = NoiseCalibrator(self.backend, shots=shots)
        self.current_noise = NoiseEstimates(0.05, 0.05)
        self.depth = config.base_depth
        
        #tracking the history
        self.noise_history = []
        self.depth_history = []
        self.energy_history = []
    
    def _update_adaptation(self):
        """Update circuit depth based on noise"""
        severity = max(self.current_noise.depolarizing_p, self.current_noise.amplitude_damping_gamma)
        
        # Reducing depth with high noise
        if severity > 0.1:
            self.depth = max(self.config.min_depth, self.depth - 1)
        elif severity < 0.05:
            self.depth = min(self.config.max_depth, self.depth + 1)
        
        self.depth_history.append(self.depth)
    
    def energy(self, param_values: List[float]) -> float:
        """Compute energy for given parameters"""
        #ansatz with current depth
        ansatz, params = hardware_efficient_ansatz(self.n, self.depth)
        
       
        param_dict = {param: val for param, val in zip(params, param_values)}
        bound_circuit = ansatz.assign_parameters(param_dict)
        
       
        estimator = AerEstimator()
        
       
        job = estimator.run([bound_circuit], [self.H])
        result = job.result()
        
        
        try:
            return float(result.values[0])
        except AttributeError:
            try:
                return float(result.data.evs[0])
            except AttributeError:
                return float(result[0])
    
    def run(self, max_outer_iters: int = 5) -> Tuple[float, List[float], List[float]]:
        
        start_time = time.time()
        
       
        self.current_noise = self.calib.estimate_noise(num_qubits=self.n)
        self.noise_history.append(self.current_noise)
        self._update_adaptation()
        
        best_val = float('inf')
        best_params = []
        all_history = []
        
        for outer in range(max_outer_iters):
           
            if outer == 0 or (outer % self.config.recalibration_frequency == 0):
                self.current_noise = self.calib.estimate_noise(num_qubits=self.n)
                self.noise_history.append(self.current_noise)
                self._update_adaptation()
            
            # Creating ansatz with current depth
            ansatz, params = hardware_efficient_ansatz(self.n, self.depth)
            
            
            init_params = np.random.uniform(-np.pi, np.pi, len(params))
            
            
            optimizer = SimpleSPSA(maxiter=20)
            
            def cost_function(param_vals):
                return self.energy(param_vals)
            
            result = optimizer.minimize(cost_function, init_params)
            
            val = result.fun
            all_history.append(val)
            self.energy_history.append(val)
            
            if val < best_val:
                best_val = val
                best_params = result.x.tolist()
        
        runtime = time.time() - start_time
        
        return best_val, best_params, all_history


#Visualizers

def run_enhanced_demo(noise_type: str = "depolarizing", noise_levels: List[float] = None):
    """Run enhanced demonstration with plots and multiple noise models"""
    
    if noise_levels is None:
        noise_levels = [0.0, 0.02, 0.05, 0.08, 0.1]
    
    print(f"🚀 Enhanced NA-VQE Demo - {noise_type.title()} Noise")
    print("=" * 50)
    
    #  Hamiltonian
    n = 4
    H = transverse_field_ising(n, h=0.8)
    
 
    results = {
        'noise_levels': [],
        'std_energies': [],
        'na_energies': [],
        'na_depths': [],
        'estimated_noise': [],
        'improvements': []
    }
    
    for noise_level in noise_levels:
        print(f"\n🔧 Testing noise level: {noise_level}")
        
        #  noise config
        noise_config = NoiseConfig(
            depolarizing_p=noise_level if noise_type in ["depolarizing", "combined"] else 0.0,
            amplitude_damping_gamma=noise_level if noise_type in ["amplitude_damping", "combined"] else 0.0,
            readout_error=noise_level if noise_type == "readout" else 0.0,
            noise_type=noise_type
        )
        
        #  noisy backend
        noise_injector = NoiseInjector(noise_config)
        noisy_backend = noise_injector.get_noisy_backend(shots=1000)
        
        # Standard VQE
        print("   Standard VQE...")
        std_vqe = StandardVQE(H, n, noisy_backend, shots=1000, depth=2)
        std_energy, _, _ = std_vqe.run(max_iterations=20)
        
        # NA-VQE
        print("   NA-VQE...")
        navqe = NAVQE(H, n, noisy_backend, shots=1000)
        na_energy, _, _ = navqe.run(max_outer_iters=3)
        
        improvement = ((std_energy - na_energy) / abs(std_energy)) * 100 if std_energy != 0 else 0
        
        print(f"    Standard VQE: {std_energy:.4f}")
        print(f"    NA-VQE: {na_energy:.4f}")
        print(f"    Improvement: {improvement:.2f}%")
        print(f"    NA-VQE depth: {navqe.depth}")
        print(f"    Estimated noise: {navqe.current_noise}")
        
        
        results['noise_levels'].append(noise_level)
        results['std_energies'].append(std_energy)
        results['na_energies'].append(na_energy)
        results['na_depths'].append(navqe.depth)
        results['estimated_noise'].append(max(navqe.current_noise.depolarizing_p, navqe.current_noise.amplitude_damping_gamma))
        results['improvements'].append(improvement)
    
    
    create_analysis_plots(results, noise_type)
    
    return results

def create_analysis_plots(results: Dict, noise_type: str):
   
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NA-VQE Analysis - {noise_type.title()} Noise', fontsize=16, fontweight='bold')
    
    # Plot 1: Energy vs Noise Level
    ax1 = axes[0, 0]
    ax1.plot(results['noise_levels'], results['std_energies'], 'o-', label='Standard VQE', 
             linewidth=2, markersize=8, color='red', alpha=0.8)
    ax1.plot(results['noise_levels'], results['na_energies'], 's-', label='NA-VQE', 
             linewidth=2, markersize=8, color='blue', alpha=0.8)
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Final Energy', fontsize=12)
    ax1.set_title('Energy vs Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Circuit Depth vs Noise Level
    ax2 = axes[0, 1]
    ax2.plot(results['noise_levels'], results['na_depths'], 'o-', 
             linewidth=2, markersize=8, color='green', alpha=0.8)
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('NA-VQE Circuit Depth', fontsize=12)
    ax2.set_title('Adaptive Circuit Depth', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_ylim(0.5, 4.5)
    
    # Plot 3: Improvement vs Noise Level
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in results['improvements']]
    bars = ax3.bar(results['noise_levels'], results['improvements'], 
                   color=colors, alpha=0.7, width=0.015)
    ax3.set_xlabel('Noise Level', fontsize=12)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('NA-VQE Performance Improvement', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
  
    for i, (bar, val) in enumerate(zip(bars, results['improvements'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Plot 4: Estimated vs Actual Noise
    ax4 = axes[1, 1]
    ax4.plot(results['noise_levels'], results['noise_levels'], '--', 
             label='Actual Noise', linewidth=2, color='black', alpha=0.7)
    ax4.plot(results['noise_levels'], results['estimated_noise'], 'o-', 
             label='Estimated Noise', linewidth=2, markersize=8, color='purple', alpha=0.8)
    ax4.set_xlabel('Actual Noise Level', fontsize=12)
    ax4.set_ylabel('Estimated Noise Level', fontsize=12)
    ax4.set_title('Noise Estimation Accuracy', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()
    
    #  summary statistics
    print(f"\n Summary Statistics for {noise_type.title()} Noise:")
    print("-" * 50)
    avg_improvement = np.mean(results['improvements'])
    max_improvement = np.max(results['improvements'])
    min_improvement = np.min(results['improvements'])
    
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"Maximum Improvement: {max_improvement:.2f}%")
    print(f"Minimum Improvement: {min_improvement:.2f}%")
    
    # Noise estimation accuracy
    noise_errors = [abs(actual - est) for actual, est in zip(results['noise_levels'], results['estimated_noise'])]
    avg_noise_error = np.mean(noise_errors)
    print(f"Average Noise Estimation Error: {avg_noise_error:.4f}")

def run_noise_model_comparison():
   
    
    noise_types = ["depolarizing", "amplitude_damping", "readout", "combined"]
    noise_levels = [0.0, 0.05, 0.1]
    
    print(" Noise Model Comparison")
    print("=" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('NA-VQE Performance Across Different Noise Models', fontsize=16, fontweight='bold')
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, noise_type in enumerate(noise_types):
        print(f"\n Testing {noise_type} noise...")
        
        results = run_enhanced_demo(noise_type, noise_levels)
        
        #  energy vs noise for each model
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        ax.plot(results['noise_levels'], results['std_energies'], 'o-', 
                label='Standard VQE', linewidth=2, markersize=6, color='red', alpha=0.7)
        ax.plot(results['noise_levels'], results['na_energies'], 's-', 
                label='NA-VQE', linewidth=2, markersize=6, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Final Energy')
        ax.set_title(f'{noise_type.title()} Noise')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
  
    print("Starting Enhanced NA-VQE Demo...")
    results = run_enhanced_demo("depolarizing")
    

    # run_noise_model_comparison()
