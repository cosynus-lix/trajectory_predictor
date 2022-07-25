import torch.nn as nn
class FeedforwardNeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNet, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim).double()

        # Input dimension
        self.input_dim = input_dim

    def forward(self, x):
        x = x.reshape(-1,self.input_dim)
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out