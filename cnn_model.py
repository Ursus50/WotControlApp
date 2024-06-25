import torch
import torch.nn as nn

class CNN_MLP(nn.Module):
    def __init__(self, input_channels, input_length, hidden_size1, hidden_size2, output_size):
        super(CNN_MLP, self).__init__()
        # Warstwa konwolucyjna 1D
        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Warstwy w pełni połączone
        conv_output_length = input_length // 2
        self.fc1 = nn.Linear(8 * conv_output_length, hidden_size1)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax na wyjściu, dim=1 dla batcha

    def forward(self, x):
        # Warstwa konwolucyjna 1D
        x = x.unsqueeze(1)  # Dodaj kanał na wejściu
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)

        # Spłaszczenie tensoru
        x = x.view(x.size(0), -1)

        # Warstwy w pełni połączone
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def load_model_cnn(model_path, input_channels, input_length, hidden_size1, hidden_size2, output_size):
    model = CNN_MLP(input_channels, input_length, hidden_size1, hidden_size2, output_size)
    model.load_state_dict(torch.load(model_path))
    return model
