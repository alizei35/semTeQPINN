import jax
import jax.numpy as jnp

def init_params(key,input_dim,output_dim,n_hidden_layers,branch_width):
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

def forward(params,x):
    for param in params[:-1]:
        x_augmented = jnp.append(x,1.0)
        x = jax.nn.tanh(jnp.dot(param,x_augmented))
    x_augmented = jnp.append(x,1.0)
    x = jnp.dot(params[-1],x_augmented)
    return x
