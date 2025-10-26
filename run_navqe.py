import warnings
warnings.filterwarnings("ignore")

from qiskit_aer import Aer

n = 4
H = transverse_field_ising(n, h=0.8)
backend = Aer.get_backend("aer_simulator")
backend.set_options(seed_simulator=1234)

navqe = NAVQE(
    H=H,
    n_qubits=n,
    backend=backend,
    shots=1000,     
    seed=1234
)

best_val, best_params, hist = navqe.run(max_outer_iters=3)
print("Estimated noise:", navqe.current_noise)
print("Chosen depth:", navqe.depth)
print("Best energy:", best_val)
print("Last few iters:", hist[-5:] if len(hist) >= 5 else hist)
