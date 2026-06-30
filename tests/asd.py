import jax
import jax.numpy as jnp
import pennylane as qp
from catalyst import qjit, for_loop, grad, vmap
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
dev = qp.device("lightning.qubit", wires=N_QUBITS)

# FNN parameter initialization function
def init_fnn_params(key,input_dim,output_dim,n_hidden_layers,branch_width):
    params = []
    keys = jax.random.split(key, n_hidden_layers+1)

    def init_layer(layer_key,in_dim,out_dim):
        w_key, b_key = jax.random.split(layer_key)
        limit = jnp.sqrt(6.0/(in_dim+out_dim))
        w = jax.random.uniform(w_key,(out_dim,in_dim),
                               minval=-limit,maxval=limit)
        b = jnp.zeros((out_dim,))
        return jnp.column_stack([w,b])

    params.append(init_layer(keys[0],input_dim,branch_width))
    for i in range(1,n_hidden_layers):
        params.append(init_layer(keys[i],branch_width,branch_width))
    params.append(init_layer(keys[-1],branch_width,output_dim))

    return params

# FNN forward function
def fnn_forward(params,x):
    for param in params[:-1]:
        x_augmented = jnp.append(x,1.0)
        x = jax.nn.tanh(jnp.dot(param,x_augmented))
    x_augmented = jnp.append(x,1.0)
    x = jnp.dot(params[-1],x_augmented)
    return x

@qjit
@qp.qnode(dev,interface="jax",diff_method="adjoint")
def circuit(x,basis,thetas):
    n_qubits = len(dev.wires)
    n_layers = thetas.shape[0]

    # FNN Basis Embedding
    qp.AngleEmbedding(features=basis*x,wires=dev.wires,rotation='Y')

    # Hardware Efficient Ansatz
    @for_loop(0,n_layers,1)
    def loop_layers(i):
        @for_loop(0,n_qubits,1)
        def loop_qubits_rot(j):
            qp.RX(thetas[i,j,0],wires=j)
            qp.RY(thetas[i,j,1],wires=j)
            qp.RZ(thetas[i,j,2],wires=j)
        loop_qubits_rot()

        @for_loop(0,n_qubits-1,1)
        def loop_qubits_cnot(j):
            qp.CNOT(wires=[j,j+1])
        loop_qubits_cnot()
    loop_layers()

    return qp.expval(qp.sum(*[qp.PauliZ(i) for i in range(n_qubits)]))

def u_single(vars, coord):
    params = vars[0]
    thetas = vars[1]
    coord_rescaled = 0.95*(2.0*coord-1.0)
    basis = fnn_forward(params,coord_rescaled)
    return circuit(coord_rescaled,basis,thetas)
du_dx_single = grad(u_single,argnums=1,method="auto")
u_batched = vmap(u_single, in_axes=(None,0))
devs = grad(du_dx_single,argnums=0,method="auto")

params = init_fnn_params(init_key,input_dim=INPUT_DIM,output_dim=N_QUBITS,
                         n_hidden_layers=FNN_HIDDEN_LAYER,
                         branch_width=FNN_BRANCH_WIDTH)
thetas = jax.random.uniform(thetas_key,(N_LAYERS,N_QUBITS,3),
                            minval=0.0,maxval=1.0)
vars = tuple([params,thetas])
coord = jnp.array([0.0])
coords = jax.random.uniform(collocation_key,(X_COLLOC_POINTS,1),
                             minval=0.0,maxval=X_END)

print(devs(vars,coord))



# def derivatives_fnc(u,x):
#     du_dx = 4*u - 6*u**2 + jnp.sin(50*x) + u*jnp.cos(25*x) - 0.5
#     return du_dx
# 
# def loss_diff_fnc(vars, coords):
#     u = u_batched(vars,coords)
#     du_dx_for = du_dx_batched(vars, coords)[:,0]
#     du_dx_ref = derivatives_fnc(u,coords[:,0])
#     res = du_dx_for - du_dx_ref
#     return jnp.mean(res**2)
# 
# def loss_bndr_fnc(vars):
#     u = u_single(vars,jnp.array([0.0]))
#     return (u-0.75)**2
# 
# def loss_fnc(vars, coords):
#     loss_diff = loss_diff_fnc(vars,coords)
#     loss_bndr = loss_bndr_fnc(vars)
#     return BOUNDARY_SCALE*loss_bndr + loss_diff
# coords = jax.random.uniform(collocation_key,(X_COLLOC_POINTS,1),
#                              minval=0.0,maxval=X_END)
# @jax.jit
# def loss_diff(vars,coords):
#     return loss_diff_fnc(vars,coords)
# print(loss_diff(vars,coords))
# 
# @jax.jit
# def loss_bndr(vars):
#     return loss_bndr_fnc(vars)
# print(loss_bndr(vars))
# 
# @jax.jit
# def loss(vars,coords):
#     return loss_fnc(vars,coords)
# print(loss(vars,coords))
# 
# 
# @jax.jit
# def loss_value_and_grad(vars,coords):
#     return loss_value_and_grad_fnc(vars,coords)

# solver = jaxopt.LBFGS(fun=loss_fnc, maxiter=100, tol=1e-9, stepsize=0.0,
#                       history_size=100, linesearch="zoom")
# state = solver.init_state(vars, coords=coords)
# @jax.jit
# def lbfgs_step(vars,state,coords):
#     vars, state = solver.update(vars,state,coords=coords)
#     return vars, state, state.value
# lbfgs_step(vars,state,coords)

# x_batch = jax.random.uniform(collocation_key,(X_COLLOC_POINTS,1),
#                              minval=0.0,maxval=X_END)
# x_batch = jnp.sort(x_batch, axis=0)
# def scipy_ode_func(x,u):
#     return 4*u - 6*u**2 + np.sin(50*x) + u*np.cos(25*x) - 0.5
# x_eval_points = np.array(x_batch[:,0])
# sol = solve_ivp(scipy_ode_func, [0.0,X_END+1e-6],[0.75],t_eval=x_eval_points)
# reference = jnp.array(sol.y[0])
# 
# params = init_fnn_params(init_key,input_dim=INPUT_DIM,output_dim=N_QUBITS,
#                          n_hidden_layers=FNN_HIDDEN_LAYER,
#                          branch_width=FNN_BRANCH_WIDTH)
# thetas = jax.random.uniform(thetas_key,(N_LAYERS,N_QUBITS,3),
#                             minval=0.0,maxval=1.0)
# vars = {"params":params,"thetas":thetas}
# 
# solver = jaxopt.LBFGS(fun=loss_fnc, maxiter=100, tol=1e-9, stepsize=0.0,
#                       history_size=100, linesearch="zoom")
# state = solver.init_state(vars, x_batch=x_batch)
# 
# @jax.jit
# def lbfgs_step(vars,state,x_batch):
#     vars, state = solver.update(vars,state,x_batch=x_batch)
#     return vars, state, state.value
# 
# @jax.jit
# def compute_MSE_ref(vars,x_batch,reference):
#     prediction = u_batched(vars,x_batch)
#     return jnp.mean((prediction-reference)**2), prediction
# 
# print("Starting L-BFGS Optimizaiton...")
# for epoch in range(100):
#     vars, state, loss_val = lbfgs_step(vars,state,x_batch)
#     mse, prediction = compute_MSE_ref(vars,x_batch,reference)
#     print(f"Epoch {epoch}, Loss: {loss_val:.2E}, MSE: {mse:.2E}")
# 
# import matplotlib.pyplot as plt
# reference_np = np.array(reference)
# prediction_np = np.array(prediction)
# fig = plt.figure()
# ax = fig.subplots()
# ax.plot(x_eval_points,reference_np,label='true')
# ax.plot(x_eval_points,prediction_np,label='appr')
# ax.legend()
# plt.show()
