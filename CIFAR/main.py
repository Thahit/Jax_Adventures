# again with the help of https://github.com/gordicaleksa/get-started-with-JAX
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax.training import train_state
import optax
import torch
from torchvision import datasets
import time

# general parameters
batch_size=256
lr = 0.2
num_epochs = 10

#___________________________data________________________________________________
def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))

    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])

    return imgs, labels

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,),
    batch_size=batch_size, shuffle=True,
    collate_fn=custom_collate_fn,
    drop_last = True # because the last one might(does) have the wrong size
    )

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, download=True,),
    batch_size=batch_size, shuffle=True,
    collate_fn=custom_collate_fn,
    drop_last = True
    )

'''for imgs, sol in train_loader:
    print(imgs.shape)
    break'''
img_size = (32, 32, 3)

#__________________________________________model________________________________________________________
class master_model(flax.linen.Module): 
    @flax.linen.compact
    def __call__(self, x):#size: 32
        x = flax.linen.Conv(features=32, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))#size: 16
        x = flax.linen.Conv(features=64, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))#size: 8
        x = flax.linen.Conv(features=128, kernel_size=(3, 3))(x)
        x = flax.linen.relu(x)
        x = flax.linen.avg_pool(x, window_shape=(2, 2), strides=(2, 2))#size: 4
        x = x.reshape((x.shape[0], -1))  # flatten == 2024
        # print(x.shape) # if not sure
        x = flax.linen.Dense(features=2024)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=256)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(features=10)(x)
        x = flax.linen.softmax(x)
        return x

#_______________________________loss_________________________________________________________
@jax.jit
def loss_fn(params, inputs, labels):
    predictions  = master_model().apply({'params': params}, inputs)# you are predicting inside the loss
    
    loss = jnp.mean(jnp.sum(labels * (1-predictions), axis=-1)) + jnp.mean(jnp.sum(jnp.abs(labels-1) * predictions, axis=-1))
    return  loss, predictions

#_________________________________________training___________________________________________
@jax.jit
def train_step(state, imgs, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, imgs, labels_onehot)
    state = state.apply_gradients(grads=grads)  
    metrics = compute_metrics(predictions=predictions, loss=loss, labels=labels)
    return state, metrics

@jax.jit
def eval_step(state, imgs, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    loss, predictions = loss_fn(state.params, imgs, labels_onehot)
    return compute_metrics(predictions=predictions, loss=loss, labels=labels)

def train_one_epoch(state, dataloader, epoch):
    """Train for 1 epoch on the training set."""
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels)
        # lower the lr with time
        state.tx.learning_rate = state.tx.learning_rate*.95
        
        batch_metrics.append(metrics)

    # Aggregate the metrics
    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np

def evaluate_model(state, dataloader):#test set
    batch_metrics = []
    for imgs, labels in dataloader:
        metrics = eval_step(state, imgs, labels)
        batch_metrics.append(metrics)
    #metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    #metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    
    #Aggregate the metrics
    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    return epoch_metrics_np



def create_train_state(key, learning_rate, momentum):
    model = master_model()
    params = model.init(key, jnp.ones([1, *img_size]))['params']
    sgd_opt = optax.sgd(learning_rate, momentum)
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=sgd_opt)

def compute_metrics(*, predictions, loss, labels):
    #one_hot_labels = jax.nn.one_hot(labels, num_classes=10)

    #loss = ... # We already computed the loss, so why do it agin?
    accuracy = jnp.mean(jnp.argmax(predictions, -1) == labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

seed = 42
momentum = 0.9

train_state = create_train_state(jax.random.PRNGKey(seed), lr, momentum)

for epoch in range(1, num_epochs + 1):
    
    start = time.time()
    train_state, train_metrics = train_one_epoch(train_state, train_loader, epoch)
    end = time.time()
    
    print(f"Train epoch: {epoch}, loss: {train_metrics['loss']:.5f}, accuracy: {train_metrics['accuracy']:.5f}, time: {end-start:.2f} seconds.")
    
    if epoch % 2 == 0:
        test_metrics = evaluate_model(train_state, test_loader)
        print(f"Test epoch: {epoch}, loss: {test_metrics['loss']:.5f}, accuracy: {test_metrics['accuracy']:.5f}")