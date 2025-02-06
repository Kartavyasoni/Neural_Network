import numpy as np

class Perceptron:
     def __init__(self, inputs, bias=1):
          self.weights = (np.random.rand(inputs+1) *2) - 1
          self.bias = bias
          
     def run(self, x):
          x_sum = np.dot(np.append(x, self.bias), self.weights)
          return self.sigmoid(x_sum)
     
     def set_weights(self, w_init):
          self.weights = np.array(w_init)
          
     def sigmoid(self, x):
          return 1/(1+np.exp(-x))
     
# Test Code OR Gate
neuron = Perceptron(inputs=2)
neuron.set_weights([15, 15, -10])

print('0 0 = {0:.10f}'.format(neuron.run([0, 0])))
print('0 1 = {0:.10f}'.format(neuron.run([0, 1])))
print('1 0 = {0:.10f}'.format(neuron.run([1, 0])))
print('1 1 = {0:.10f}'.format(neuron.run([1, 1])))