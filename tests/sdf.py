import pennylane as qml
from jax import numpy as np
import catalyst

print("PennyLane version:", qml.__version__)
print("Catalyst version :", catalyst.__version__)

# Hyperparameters
compiler="catalyst"
device="lightning.qubit"
n_wires = 22

# Sample
data = qml.numpy.random.rand(n_wires)

# Weights
weights = np.ones([n_wires])# Device

dev = qml.device(device, wires=n_wires)

# Quantum circuit
@qml.qnode(dev, interface="jax", diff_method="adjoint")
def circuit(data, weights):

    for i in range(n_wires):
        qml.RY(data[i], wires=i)
        qml.RX(weights[i], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

# JIT circuit
circuit = qml.qjit(circuit, compiler=compiler)

# Loss
def loss_fn(weights, data):
    predictions = np.array(circuit(data, weights))
    loss = np.average((data - predictions) ** 2)
    return loss

# Loss Gradient
grad = qml.qjit(catalyst.grad(loss_fn, method="auto"))

# Compute Grad
print(grad(weights, data))
