# output = w*x + b
# output = activation_function(output)
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# sofmax
output = torch.softmax(x, dim=0)
print('\nSoftmax from torch api: \n\n', output)
sm = nn.Softmax(dim=0)
output = sm(x)
print('\nSoftmax from nn module: \n\n', output)

# sigmoid 
output = torch.sigmoid(x)
print('\nSigmoid from torch api: \n\n', output)
s = nn.Sigmoid()
output = s(x)
print('\nSigmoid from nn module: \n\n', output)

#tanh
output = torch.tanh(x)
print('\nTanH from torch api: \n\n', output)
t = nn.Tanh()
output = t(x)
print('\nTanH from nn module: \n\n', output)

# relu
output = torch.relu(x)
print('\nReLU from torch api: \n\n', output)
relu = nn.ReLU()
output = relu(x)
print('\nReLU from nn module: \n\n', output)

# leaky relu
output = F.leaky_relu(x)
print('\nLeaky ReLU from torch api: \n\n', output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print('\nLeaky ReLU from nn module: \n\n', output)

#nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.
#torch.relu on the other side is just the functional API call to the relu function,
#so that you can add it e.g. in your forward method yourself.

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out