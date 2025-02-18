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

class MultiLayerPerceptron:
     def __init__(self, layers, bias=1):
          self.layers = np.array(layers, dtype=object)
          self.bias = bias
          self.network = []
          self.values = []
          
          for i in range(len(self.layers)):
               self.values.append([])
               self.network.append([])
               self.values[i] = [0.0 for _ in range(self.layers[i])]
               if i>0:
                    for _ in range(self.layers[i]):
                         self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))
                         
          self.network = np.array([np.array(i) for i in self.network], dtype=object)
          self.values = np.array([np.array(i) for i in self.values], dtype=object)
               
     def set_weights(self, w_init):
          for i in range(len(w_init)):
               for j in range(len(w_init[i])):
                    self.network[i+1][j].set_weights(w_init[i][j])
          
     def printWeights(self):
          for i in range(1,len(self.network)):
               for j in range(self.layers[i]):
                    print('Layer:', i+1, 'Neuron:', j, self.network[i][j].weights)
          
     def run(self, x):
          x = np.array(x, dtype=object)
          self.values[0] = x
          for i in range(1, len(self.network)):
               for j in range(self.layers[i]):
                    self.values[i][j] = self.network[i][j].run(self.values[i-1])
          return self.values[-1]
     
     
# Test Code XOR Gate
mlp = MultiLayerPerceptron(layers=[2, 2, 1])
mlp.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])
mlp.printWeights()
print('MLP:')

print('0 0 = {0:.10f}'.format(mlp.run([0, 0])[0]))
print('0 1 = {0:.10f}'.format(mlp.run([0, 1])[0]))
print('1 0 = {0:.10f}'.format(mlp.run([1, 0])[0]))
print('1 1 = {0:.10f}'.format(mlp.run([1, 1])[0]))