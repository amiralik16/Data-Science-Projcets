import numpy as np
from sklearn.metrics import mean_squared_error
# Possible bug, not passed by reference
def basic_func(x):
    return sum(x)

def sigmoid(x):

    return 1/(1+ np.exp(-x))

def sigmoid_d(x):

    return sigmoid(x) * sigmoid(1 - x)

def relu_d(x):
    # print(1. * (x > 0))
    return 1 * (x > 0)



def relu(x):
    # print(x*(x>0))
    return x * (x > 0)

class Neuron:

    def __init__(self, input_dim, func=sigmoid, func_d = sigmoid_d):
        self.inputs = np.random.rand(1, input_dim)
        self.desired_inputs = np.random.rand(1, input_dim)
        self.weights = np.random.rand(1,input_dim)
        self.func = func
        self.func_d = func_d
        self.output = np.random.normal()
        self.bias = np.random.normal()
        self.desired_output = np.random.normal()
        # print(self.output)
    
    def update_input(self,neurons):
        '''
        neurons is a list of neuron objects that this needs to be connected to
        '''
        self.inputs = [neuron.output for neuron in neurons]

    def forward(self):
        self.output = self.func(np.vdot(self.weights,self.inputs) + self.bias)
        # print(self.output)
    
    def back_prop(self):
        print('\n')
        # print(self.output)
        # print(self.desired_output)
        d_weights = np.dot(self.inputs,self.func_d(np.vdot(self.weights,self.inputs) + self.bias)*2*(self.output-self.desired_output))
        # print(d_weights)
        self.weights = self.weights + d_weights
        print(f'd_weights are {d_weights}')
        print(f'd_outputs are {self.output-self.desired_output}')
        print(f'd_output is {self.desired_output}')
        d_inputs = np.dot(self.weights,self.func_d(np.vdot(self.weights,self.inputs) + self.bias)*2*(self.output-self.desired_output))
        
        print(f'weights are {self.weights}')
        print(f'inputs are {self.inputs}')
        print(f'd_inputs are {d_inputs}')
        self.desired_inputs = self.inputs + d_inputs

    def update_desired_input(self,x):
        self.desired_inputs = x

    def update_desired_output(self,x):
        self.desired_output = x



class Layer:
    def __init__(self, input_dim, num_of_neurons, func=sigmoid, func_d = sigmoid_d):
        self.inputs = np.random.rand(1, input_dim)
        self.func = func
        self.neurons = []
        self.num_of_neurons = num_of_neurons
        for i in range(num_of_neurons):
            self.neurons.append(Neuron(input_dim, func, func_d=func_d))

    def update_input(self, prev_layer):
        for neuron in self.neurons:
            neuron.update_input(prev_layer.neurons)

    def update_input_first(self, inputs):
        for neuron in self.neurons:
            neuron.inputs = inputs

    def forward(self,prev_layer):
        self.update_input(prev_layer)
        for neuron in self.neurons:
            neuron.forward()

    def forward_first(self,inputs):
        self.update_input_first(inputs)
        for neuron in self.neurons:
            neuron.forward()

    def back_prop(self, next_layer):
        self.update_desired_output(next_layer)
        for neuron in self.neurons:
            neuron.back_prop()

    def back_prop_last(self, last):
        self.update_desired_output_last(last)
        for neuron in self.neurons:
            neuron.back_prop()

    def update_desired_output(self,next_layer):
        for i,neuron in enumerate(self.neurons):
            s=0
            for next_neuron in next_layer.neurons:
                s = s + next_neuron.desired_inputs[0][i]
                # print(next_neuron.desired_inputs)
            neuron.desired_output = s/next_layer.num_of_neurons
            # print(neuron.desired_output)

    def update_desired_output_last(self,last):
        for i,neuron in enumerate(self.neurons):
            neuron.desired_output = last[i]
            # print(neuron.desired_output)


class Neural_Network:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.output = np.random.rand(len(y))
        self.layers = []
        self.active_layer = 0

    def add_layer(self,input_dim, num_of_neurons, func=sigmoid, func_d = sigmoid_d):
        self.layers.append(Layer(input_dim,num_of_neurons,func,func_d))
        
    def forward(self):
        for i,layer in enumerate(self.layers):
            self.active_layer = i
            try: 
                print('forward')
                layer.forward(self.layers[i-1])
            except: 
                layer.forward_first(self.input)   
        self.output = [neuron.output for neuron in self.layers[-1].neurons]

    def back_prop(self):
        for i,layer in enumerate(self.layers):
            self.active_layer = i
            print(f"Currently on layer:{i}\n")
            try: 
                # print('hi')
                layer.back_prop(self.layers[i+1])
            except: 
                layer.back_prop_last(self.y)   

    def train(self,n):
        for i in range(n):
            self.back_prop()
            self.forward()
            # print(self.y)
            # print('\n')
            # print(self.output)
            # break
            # print(mean_squared_error(self.y,1-np.array(self.output)))
            # print(self.output)
        print(1-np.array(self.output))


if __name__ == "__main__":
    x = [1,0,1,1,1,1]
    y = [0,1,0.2,0,1,1]

    NN = Neural_Network(x,y)
    NN.add_layer(6,6,func=relu,func_d=relu_d)
    # NN.add_layer(6,6,func=sigmoid,func_d=sigmoid_d)
    NN.train(10)
