import jax
import jaxopt
import jax.numpy as jnp
import pennylane as qp
import numpy as np
from scipy.integrate import solve_ivp
from sim_sci_sem.nn import init_fnn_params, fnn_forward
jax.config.update("jax_enable_x64", True)

INPUT_DIM = 1
N_QUBITS = 6
N_LAYERS = 5
X_END = 1.0
X_COLLOC_POINTS = 100
BOUNDARY_SCALE = 10e1
FNN_BRANCH_WIDTH = 5
FNN_HIDDEN_LAYER = 1

collocation_key,thetas_key,init_key = jax.random.split(jax.random.PRNGKey(35),3)
dev = qp.device("default.qubit", wires=N_QUBITS)

@qp.qnode(dev,interface="jax",diff_method="backprop")
def circuit(x,basis,thetas):
    n_qubits = len(dev.wires)
    n_layers = thetas.shape[0]

    # FNN Basis Embedding
    qp.AngleEmbedding(features=basis*x,wires=dev.wires,rotation='Y')

    # Hardware Efficient Ansatz
    for i in range(n_layers):
        for j in range(n_qubits):
            qp.RX(thetas[i,j,0],wires=j)
            qp.RY(thetas[i,j,1],wires=j)
            qp.RZ(thetas[i,j,2],wires=j)
        for j in range(n_qubits-1):
            qp.CNOT(wires=[j,j+1])

    return [qp.expval(qp.PauliZ(i)) for i in dev.wires]

def u_single(vars, coord):
    params = vars["params"]
    thetas = vars["thetas"]

    coord_rescaled = 0.95*(2.0*coord-1.0)

    basis = fnn_forward(params,coord_rescaled)
    expvals = circuit(coord_rescaled,basis,thetas)

    return jnp.sum(jnp.array(expvals))
du_dx_single = jax.grad(u_single,argnums=1)

u_batched = jax.vmap(u_single, in_axes=(None,0))
du_dx_batched = jax.vmap(du_dx_single, in_axes=(None,0))

def derivatives_fnc(u,x):
    du_dx = 4*u - 6*u**2 + jnp.sin(50*x) + u*jnp.cos(25*x) - 0.5
    return du_dx

def loss_diff_fnc(vars, x_batch):
    u = u_batched(vars,x_batch)
    du_dx_for = du_dx_batched(vars, x_batch)[:,0]
    du_dx_ref = derivatives_fnc(u,x_batch[:,0])
    res = du_dx_for - du_dx_ref
    return jnp.mean(res**2)

def loss_bndr_fnc(vars):
    u = u_single(vars,jnp.array([0.0]))
    return (u-0.75)**2

def loss_fnc(vars, x_batch):
    loss_diff = loss_diff_fnc(vars,x_batch)
    loss_bndr = loss_bndr_fnc(vars)
    return BOUNDARY_SCALE*loss_bndr + loss_diff

x_batch = jax.random.uniform(collocation_key,(X_COLLOC_POINTS,1),
                             minval=0.0,maxval=X_END)
x_batch = jnp.sort(x_batch, axis=0)
def scipy_ode_func(x,u):
    return 4*u - 6*u**2 + np.sin(50*x) + u*np.cos(25*x) - 0.5
x_eval_points = np.array(x_batch[:,0])
sol = solve_ivp(scipy_ode_func, [0.0,X_END+1e-6],[0.75],t_eval=x_eval_points)
reference = jnp.array(sol.y[0])

params = init_fnn_params(init_key,input_dim=INPUT_DIM,output_dim=N_QUBITS,
                         n_hidden_layers=FNN_HIDDEN_LAYER,
                         branch_width=FNN_BRANCH_WIDTH)
thetas = jax.random.uniform(thetas_key,(N_LAYERS,N_QUBITS,3),
                            minval=0.0,maxval=1.0)
vars = {"params":params,"thetas":thetas}

solver = jaxopt.LBFGS(fun=loss_fnc, maxiter=100, tol=1e-9, stepsize=0.0,
                      history_size=100, linesearch="zoom")
state = solver.init_state(vars, x_batch=x_batch)

@jax.jit
def lbfgs_step(vars,state,x_batch):
    vars, state = solver.update(vars,state,x_batch=x_batch)
    return vars, state, state.value

@jax.jit
def compute_MSE_ref(vars,x_batch,reference):
    prediction = u_batched(vars,x_batch)
    return jnp.mean((prediction-reference)**2), prediction

print("Starting L-BFGS Optimizaiton...")
for epoch in range(100):
    vars, state, loss_val = lbfgs_step(vars,state,x_batch)
    mse, prediction = compute_MSE_ref(vars,x_batch,reference)
    print(f"Epoch {epoch}, Loss: {loss_val:.2E}, MSE: {mse:.2E}")

import matplotlib.pyplot as plt
reference_np = np.array(reference)
prediction_np = np.array(prediction)
fig = plt.figure()
ax = fig.subplots()
ax.plot(x_eval_points,reference_np,label='true')
ax.plot(x_eval_points,prediction_np,label='appr')
ax.legend()
plt.show()
