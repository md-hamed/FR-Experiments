import network
from network import Network
from network import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer, ReLU
import read_data

training_data, validation_data, test_data = network.load_data_shared();

mini_batch_size = 3

net = Network([
      ConvPoolLayer(image_shape=(mini_batch_size, 1, 200, 200),
                    filter_shape=(20, 1, 5, 5), 
                    poolsize=(2, 2), 
                    activation_fn=ReLU),
      ConvPoolLayer(image_shape=(mini_batch_size, 20, 98, 98), 
                    filter_shape=(40, 20, 5, 5), 
                    poolsize=(2, 2), 
                    activation_fn=ReLU),
      FullyConnectedLayer(
          n_in=40*47*47, n_out=300, activation_fn=ReLU, p_dropout=0.3),
      FullyConnectedLayer(
          n_in=300, n_out=300, activation_fn=ReLU, p_dropout=0.5),
      SoftmaxLayer(n_in=300, n_out=423, p_dropout=0.5)], 
      mini_batch_size)
net.SGD(training_data, 10, mini_batch_size, 0.0003, 
          validation_data, test_data)
