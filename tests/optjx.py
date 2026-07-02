import jax
import time
import optax
import numpy as np
import pennylane as qp
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from sim_sci_sem.nn import linear
from sim_sci_sem.quantum import embedding, ansatz
jax.config.update("jax_enable_x64", True)

## Problem
def derivatives_fn(u,x):
    du_dx = 4*u - 6*u**2 + jnp.sin(50*x) + u*jnp.cos(25*x) - 0.5
    return du_dx

#### Solver Config ####
INPUT_DIM = 1
N_QUBITS = 6
N_LAYERS = 5
X_END = 1.0
X_COLLOC_POINTS = 100
BOUNDARY_SCALE = 10e1
FNN_BRANCH_WIDTH = 5
FNN_HIDDEN_LAYER = 1
#######################

colloc_key,init_key,thetas_key = jax.random.split(jax.random.PRNGKey(35),3)
xs = jax.random.uniform(colloc_key,(X_COLLOC_POINTS,1),minval=0.0,maxval=X_END)
xs = jnp.sort(xs,axis=0)

dev = qp.device("default.qubit", wires=N_QUBITS)
@qp.qnode(dev,interface="jax",diff_method="backprop")
def circuit(x,basis,thetas):
    n_wires = len(dev.wires)
    n_layers = thetas.shape[0]

    embedding.Trainable(basis,x,dev.wires)
    ansatz.HardwareEfficient(thetas,dev.wires,n_wires,n_layers)

    return qp.expval(qp.sum(*[qp.PauliZ(i) for i in range(n_wires)]))

def u_fn(vars,x):
    params = vars["params"]
    thetas = vars["thetas"]
    x_rescaled = 0.95*(2.0*x-1.0)
    basis = linear.forward(params,x_rescaled)
    return circuit(x_rescaled,basis,thetas)
us_fn = lambda vars: jax.vmap(u_fn,in_axes=(None,0))(vars,xs)
du_dx_fn = jax.grad(u_fn,argnums=1)
du_dxs_fn = lambda vars: jax.vmap(du_dx_fn,in_axes=(None,0))(vars,xs)

def loss_diff_fn(vars):
    us = us_fn(vars)
    du_dxs = du_dxs_fn(vars)[:,0]
    du_dxs_ref = derivatives_fn(us,xs[:,0])
    res = du_dxs - du_dxs_ref
    return jnp.mean(res**2)

def loss_bndr_fn(vars):
    u = u_fn(vars,jnp.array([0.0]))
    return (u-0.75)**2

def loss_fn(vars):
    loss_diff = loss_diff_fn(vars)
    loss_bndr = loss_bndr_fn(vars)
    return BOUNDARY_SCALE*loss_bndr + loss_diff

params = linear.init_params(init_key,input_dim=INPUT_DIM,output_dim=N_QUBITS,
                            n_hidden_layers=FNN_HIDDEN_LAYER,
                            branch_width=FNN_BRANCH_WIDTH)
thetas = jax.random.uniform(thetas_key,(N_LAYERS,N_QUBITS,3),
                            minval=0.0,maxval=1.0)
vars = {"params":params,"thetas":thetas}

## Optimization
linesearch_cfg = optax.scale_by_zoom_linesearch(
    max_linesearch_steps=25,
    max_learning_rate=1.0,
    slope_rtol=1e-4,
    curv_rtol=0.9,
    initial_guess_strategy="one",
    verbose=False
)
opt = optax.lbfgs(
    memory_size=100,
    scale_init_precond=True,
    linesearch=linesearch_cfg
)

@jax.jit
def run_opt(init_vars,max_iter,tol):
    def step(carry):
        vars,state,_ = carry
        value,grad = jax.value_and_grad(loss_fn)(vars)
        updates,state = opt.update(
            grad,state,vars,value=value,grad=grad,value_fn=loss_fn
        )
        vars = optax.apply_updates(vars,updates)
        return vars,state,value
    def continuing_criterion(carry):
        _,state,_ = carry
        iter_num = optax.tree.get(state,'count')
        grad = optax.tree.get(state,'grad')
        err = optax.tree.norm(grad)
        return (iter_num==0) | ((iter_num<max_iter) & (err>=tol))
    init_carry = (init_vars,opt.init(init_vars),float('inf'))
    final_vars,final_state,final_value = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_vars,final_state,final_value

start_time = time.perf_counter()

vars,state,value = run_opt(vars,2000,1e-7)
print(f"Loss: {value}")

value.block_until_ready()
compile_and_run_time = time.perf_counter() - start_time
print(f"The optimization took: {compile_and_run_time:.4f} seconds")

## Plotting Data
def scipy_ode_func(x,u):
    return 4*u - 6*u**2 + np.sin(50*x) + u*np.cos(25*x) - 0.5
coords_eval = np.array(xs[:,0])
sol = solve_ivp(scipy_ode_func, [0.0,X_END+1e-6],[0.75],t_eval=coords_eval)
reference = jnp.array(sol.y[0])
prediction = jax.jit(us_fn)(vars)

## Plotting
import matplotlib.pyplot as plt
reference_np = np.array(reference)
prediction_np = np.array(prediction)
fig = plt.figure()
ax = fig.subplots()
ax.plot(coords_eval,reference_np,label='true')
ax.plot(coords_eval,prediction_np,label='appr')
ax.legend()
plt.show()
