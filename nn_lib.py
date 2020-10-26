import numpy as np
import pickle
import math


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)

    def __str__(self):
      return "MSELossLayer"


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)

    def __str__(self):
      return "CrossEntropyLossLayer"


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def vectorized_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        a = self.vectorized_sigmoid(x)
        self._cache_current = a
        return a

    def backward(self, grad_z):
        a = self._cache_current
        delta_g = np.multiply(a, 1- a)

        return np.multiply(grad_z, delta_g)

    def __str__(self):
      return "SigmoidLayer"


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        self._cache_current = x
        return np.maximum(x, 0, x)

    def vectorized_ones_switch(self, g):
        return g != 0

    def backward(self, grad_z):
        g = self._cache_current
        delta_g = np.maximum(g, 0, g)
        ones = self.vectorized_ones_switch(delta_g)
        return np.multiply(ones, grad_z)

    def __str__(self):
      return "ReluLayer"

class IdentityLayer(Layer):
    """
    IdentityLayer: simple layer just emits the output of the layer before going forward,
                   and loss gradient of next layer backwards
    """

    def __init__(self):
      pass

    def forward(self, x):
      return x

    def backward(self, grad_z):
      return grad_z

    def __str__(self):
      return "IdentityLayer"


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        self._W = xavier_init([n_in, n_out])
        self._b = xavier_init([1, n_out])

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None


    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """

        z = x.dot(self._W) + self._b
        self._cache_current = x
        return z

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """

        self._grad_W_current = self._cache_current.transpose().dot(grad_z)

        self._grad_b_current = np.sum(grad_z, axis = 0)

        return grad_z.dot(self._W.transpose())

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W = np.subtract(self._W, self._grad_W_current * learning_rate)
        self._b = np.subtract(self._b, self._grad_b_current * learning_rate)
        # self._W -= self._grad_W_current * learning_rate
        # self._b -= self._grad_b_current * learning_rate


    def __str__(self):
      return (f"LinearLayer: [{self.n_in}, {self.n_out}] ")


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        self._layers = self.build_layers(input_dim, neurons, activations)


    def build_layers(self, input_dim, neurons, activations):
        assert len(neurons) > 0
        assert len(neurons) == len(activations)

        layers = []

        for index, (dimension, activation) in enumerate(zip(neurons, activations)):
          layers.append(LinearLayer(input_dim, dimension))
          layers.append(self.layer_switch(activation))
          input_dim = dimension

        return layers


    def layer_switch(self, activation_str):
        switch = {
          'identity': IdentityLayer(),
          'sigmoid': SigmoidLayer(),
          'relu': ReluLayer(),
          'cross_entropy': CrossEntropyLossLayer(),
          'mse': MSELossLayer()
        }
        return switch.get(activation_str)


    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        data = x
        for layer in self._layers:
          data = layer(data)

        return data

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        grad = grad_z
        for layer in reversed(self._layers):
          grad = layer.backward(grad)


    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for layer in self._layers:
          if (str(layer).split(" ")[0] == "LinearLayer:"):
            layer.update_params(learning_rate)


    def __str__(self):
      # allows printing of multilayer network for visualization
      layer_str =  "MultiLayerNetwork: \n"
      for layer in self._layers:
        layer_str += str(layer) + '\n'

      return layer_str

def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        assert loss_fun in ['mse', 'cross_entropy']
        self._loss_layer = MSELossLayer() if (loss_fun == 'mse') else CrossEntropyLossLayer()


    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        assert len(input_dataset) == len(target_dataset)
        p = np.random.permutation(len(input_dataset))
        return input_dataset[p], target_dataset[p]

    def batch_split(self, arr):
      data_length = len(arr)
      batches = data_length // self.batch_size
      batch_lim = batches * self.batch_size
      overflow = arr[batch_lim:]

      if (data_length - batch_lim) > 0:
        arr = np.split(arr[:-(data_length-batch_lim)], batches)
        arr.append(overflow)
      else:
        arr = np.split(arr, batches)

      return arr

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        assert len(input_dataset) == len(target_dataset)

        for epoch in range(0, self.nb_epoch):

          if self.shuffle_flag:
            input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)

          input_batches = self.batch_split(input_dataset)
          target_batches = self.batch_split(target_dataset)

          for input_batch, target_batch in zip(input_batches, target_batches):
            output = self.network.forward(input_batch)
            self._loss_layer.forward(output, target_batch)

            loss_grad = self._loss_layer.backward()

            self.network.backward(loss_grad)
            self.network.update_params(self.learning_rate)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        output = self.network.forward(input_dataset)
        loss_arr = self._loss_layer.forward(output, target_dataset)
        return np.sum(loss_arr)


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        self.mins = data.min(axis=0)
        self.maxs = data.max(axis=0)


    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        return (data - self.mins) / (self.maxs - self.mins)

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        return (data * (self.maxs - self.mins)) + self.mins


def example_main():

    input_dim = 4
    neurons = [8, 8, 3]
    activations = ["relu", "sigmoid", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
