# with the help of: https://github.com/gordicaleksa/get-started-with-JAX
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from torchvision import datasets, transforms
import torch

from jax import grad, jit, vmap, value_and_grad

lr = 0.001
batch_size = 512
num_epochs = 300

def init_mlp_params(layer_widths):
    params = []

    # Allocate weights and biases (model parameters)
    # Notice: we're not using JAX's PRNG here - doesn't matter for this simple example
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
                biases=np.ones(shape=(n_out,))
        )
    )
    return params

# Instantiate a single input - single output, 3 layer (2 hidden layers) deep MLP
params = init_mlp_params([3*(32**2), 512*2, 256, 10])

# Another example of how we might use tree_map - verify that shapes make sense:
tree = jax.tree_map(lambda x: x.shape, params)
#print(tree)

def forward(params, x):
    *hidden, last = params

    for layer in hidden:
        x = jax.nn.relu(jnp.dot(x, layer['weights']) + layer['biases'])
    last_layer = jnp.dot(x, last['weights']) + last['biases']
    #return last_layer - jax.scipy.special.logsumexp(last_layer)
    return jax.nn.softmax(last_layer)

batched_predict = vmap(forward, in_axes=(None, 0))

@jit
def loss_fn(params, x, y):
    predictions  = batched_predict(params, x)
    #return -jnp.mean(predictions * y)
    return  jnp.mean(jnp.sum(y * (1-predictions), axis=-1)) +jnp.mean(jnp.sum(jnp.abs(y-1) * predictions, axis=-1))

'''
@jit  # notice how we do jit only at the highest level - XLA will have plenty of space to optimize
def update(params, x, y):

    # Note that grads is a pytree with the same structure as params.
    # grad is one of the many JAX functions that has built-in support for pytrees!
    grads = jax.grad(loss_fn)(params, x, y)

    # Task: analyze grads and make sure it has the same structure as params

    # SGD update
    return jax.tree_util.tree_map(#jax.tree_multimap
        lambda p, g: p - lr * g, params, grads  # for every leaf i.e. for every param of MLP
    )

xs = np.random.normal(size=(128, 1))
ys = xs ** 2  # let's learn how to regress a parabola

# Task experiment a bit with other functions (polynomials, sin, etc.)

num_epochs = 5000
for _ in range(num_epochs):
    params = update(params, xs, ys)  # again our lovely pattern

plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label='Model prediction')
plt.legend()
plt.show()
'''

'''
#test how convolutions could work
def init_conv_params(kernel_size:array, kernel_nr: array):#arrays need to have same lengths and len=nr layers
    params = []
    last_size = 3#color channels

    for size, nr in zip(kernel_size, kernel_nr):
        params.append(
            dict(#name = "conv",
                    kernels=np.random.normal(size=(nr, last_size, size, size)),
                    #padding="same",
                    #stride=(1,1)
                )
        )
        last_size = nr
    
    return params
params = init_conv_params([5,5,3],[16,32,32])

params.extend(init_mlp_params([128,10]))#might ned to adjust size
#print(params)
tree = jax.tree_map(lambda x: x.shape, params)
print(tree)
'''

'''
def forward(params, nr_cnn, x):#change
    for i in range(nr_cnn):
        layer = params[i]
        #ccn stuff
        y=[]
        for kernel in layer["kernels"]:
            kernel=jnp.reshape(kernel,[1, kernel.shape[0],kernel.shape[1],kernel.shape[2]])
            print("x shape: ", x.shape, "\tkernel shape:", kernel.shape)
            y.append(lax.conv(x, kernel, window_strides = (1,1), padding='same'))
        x=jnp.array(y)
        print(x.shape)
        break
    #flatten
    
    #linear layers
    for i in range(nr_cnn, len(params)-1):
        layer = params[i]
        x = jax.nn.relu(jnp.dot(x, layer['weights']) + layer['biases'])

    return jnp.dot(x, params[-1]['weights']) + params[-1]['biases']
'''

def numpy_transformer(x):
    return np.array(x)

def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])

    return imgs, labels

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform = numpy_transformer
                   ),
    batch_size=batch_size, shuffle=True,
    collate_fn=custom_collate_fn,
    drop_last = True
    )

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=True,
                   transform = numpy_transformer
                   ),
    batch_size=batch_size, shuffle=True,
    collate_fn=custom_collate_fn,
    drop_last = True
    )

'''for batch_images, labels in train_loader:
    labels = jax.nn.one_hot(labels, 10)
    print(batch_images.shape)
    break'''

@jit
def update(params, imgs, gt_lbls, lr=0.001):
    loss, grads = value_and_grad(loss_fn)(params, imgs, gt_lbls)
    
    return loss, jax.tree_map(lambda p, g: p - lr*g, params, grads)

def minitest(params, test_loader):
    accs = []
    for idx, (imgs, labels) in enumerate(test_loader):
        flat_imgs = jnp.reshape(imgs,(batch_size,-1))
        predictions = batched_predict(params, flat_imgs)
        predicted_class =  np.argmax(predictions, axis=1)
        accs.append(jnp.mean(predicted_class == labels))
        
        if idx > 10:
            break
    return jnp.mean(jnp.array(accs))

for epoch in range(num_epochs):

    for idx, (imgs, labels) in enumerate(train_loader):
        gt_labels = jax.nn.one_hot(labels, 10)
        flat_imgs = jnp.reshape(imgs,(batch_size,-1))
        loss, params = update(params, flat_imgs, gt_labels)
        
    if epoch % 10 == 0:
        acc = minitest(params, test_loader)
        print("epoch: ", epoch, "\tLoss(last batch): ", loss, "\t test accuracy: ", acc)