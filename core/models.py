import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import LSTM, GRU, functional as F
import sys
from os import path

# Add the parent directory to the PYTHONPATH
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from core.common import BATCH_SIZE, FRAMES, FEATURES, OBJ_CUDA


class Net(nn.Module):

    def __init__(self, large=True, lstm=True):
        super(Net, self).__init__()

        self.large = large
        self.lstm = lstm
        self.relu = nn.ReLU()

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(input_size=FEATURES, hidden_size=FRAMES, num_layers=1, batch_first=True)
        else:
            self.rnn = GRU(input_size=FEATURES, hidden_size=FRAMES, num_layers=1, batch_first=True)

        if large:
            self.lin1 = nn.Linear(FRAMES ** 2, 26)
            self.lin2 = nn.Linear(26, 2)
        else:
            self.lin = nn.Linear(FRAMES ** 2, 2)

        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = Variable(torch.zeros(1, BATCH_SIZE, FRAMES))
        c = Variable(torch.zeros(1, BATCH_SIZE, FRAMES))

        if OBJ_CUDA:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):
        if OBJ_CUDA:
            self.rnn.flatten_parameters()

        # (batch, frames, features)
        if hasattr(self, 'lstm') and self.lstm:
            if x.device.type != self.hidden[0].device.type:
                hidden = [tensor.to("cpu") for tensor in self.hidden]
                x, _ = self.rnn(x, hidden)
            else:
                x, _ = self.rnn(x, self.hidden)

        else:
            x, _ = self.rnn(x)

        x = x.contiguous().view(-1, FRAMES ** 2)

        # (batch, units)
        if self.large:
            x = self.relu(self.lin1(x))
            x = self.lin2(x)
        else:
            x = self.lin(x)

        return self.softmax(x)


class BiRNN(nn.Module):
    '''
    Bi-directional layer of gated recurrent units.
    Includes a fully connected layer to produce binary output.
    '''

    def __init__(self, num_in, num_hidden, batch_size=BATCH_SIZE, large=True, lstm=False, fcl=True, bidir=False):
        super(BiRNN, self).__init__()

        self.num_hidden, self.batch_size = num_hidden, batch_size
        self.lstm, self.bidir, self.layers = lstm, bidir, 2 if large else 1

        if lstm:
            self.hidden = self.init_hidden()
            self.rnn = LSTM(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18 if large else 16
        else:
            self.rnn = GRU(num_in, num_hidden, num_layers=self.layers, bidirectional=self.bidir, batch_first=True)
            sz = 18

        embed_sz = num_hidden * 2 if self.bidir or self.layers > 1 else num_hidden

        if not fcl:
            self.embed = nn.Linear(embed_sz, 2)
        else:
            if large:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz + 14),
                    nn.BatchNorm1d(sz + 14),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz + 14, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )
            else:
                self.embed = nn.Sequential(
                    nn.Linear(embed_sz, sz),
                    nn.BatchNorm1d(sz),
                    nn.Dropout(p=0.2),
                    nn.ReLU(),
                    nn.Linear(sz, 2)
                )

    def init_hidden(self):
        num_dir = 2 if self.bidir or self.layers > 1 else 1
        h = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))
        c = Variable(torch.zeros(num_dir, self.batch_size, self.num_hidden))

        if OBJ_CUDA:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):
        if OBJ_CUDA:
            self.rnn.flatten_parameters()

        x = x.permute(0, 2, 1)

        if self.lstm:
            x, self.hidden = self.rnn(x, self.hidden)
        else:
            x, self.hidden = self.rnn(x)

        # Extract outputs from forward and backward sequence and concatenate
        # If not bidirectional, only use last output from forward sequence
        x = self.hidden.view(self.batch_size, -1)

        # (batch, features)
        return self.embed(x)


class GatedConv(nn.Module):
    '''
    Gated convolutional layer using tanh as activation and a sigmoidal gate.
    The convolution is padded to keep its original dimensions.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=True):
        super(GatedConv, self).__init__()

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )
        self.conv_gate = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x) * self.conv_gate(x)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=True):
        super(Conv, self).__init__()

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class GatedResidualConv(nn.Module):
    '''
    Legacy class.
    Gated residual convolutional layer using tanh as activation and a sigmoidal gate.
    Outputs the accumulated input to be used in the following layer, as well as a
    residual connection that is added to the output of the following layer using
    element-wise multiplication. Input and output sizes are unchanged.
    '''

    def __init__(self, channels, kernel_size=3, dilation=1):
        super(GatedResidualConv, self).__init__()

        self.gated_conv = GatedConv(channels, channels)

    def forward(self, x, r=None):
        # Residual connection defaults to x
        if r is None:
            r = x
        out = self.gated_conv(x)

        # (acummulated input, residual connection)
        return out * x, out * r


class NickNet(nn.Module):
    '''
    This network consists of (gated) convolutional layers,
    followed by a bi-directional recurrent layer and one or
    more fully connected layers. Output is run through a
    softmax function.
    '''

    def __init__(self, large=True, residual_connections=False, gated=True, lstm=False,
                 fcl=True, bidir=False, frames=FRAMES, features=FEATURES):

        super(NickNet, self).__init__()

        self.large = large
        self.residual_connections = residual_connections

        # Define number of channels depending on configuration.
        # This is done to ensure that number of parameters are
        # held some-what constant for all model configurations.
        if large:
            if gated:
                conv_channels1, conv_channels2, conv_channels3, conv_channels4 = 32, 28, 25, 18
            else:
                conv_channels1, conv_channels2, conv_channels3, conv_channels4 = 38, 35, 31, 24
            conv_channels_out = conv_channels4
        else:
            if gated:
                conv_channels1, conv_channels2, conv_channels3 = 20, 18, 16
            else:
                conv_channels1, conv_channels2, conv_channels3 = 26, 20, 16
            conv_channels_out = conv_channels3

        # Gated convolution with residual connections
        if residual_connections:
            conv_channels3 = conv_channels2
            self.conv1 = GatedConv(features, conv_channels3)
            self.conv2 = GatedResidualConv(conv_channels3)
            self.conv3 = GatedResidualConv(conv_channels3)
            if large:
                self.conv4 = GatedResidualConv(conv_channels3)

        # Gated convolution
        elif gated:
            self.conv1 = GatedConv(features, conv_channels1)
            self.conv2 = GatedConv(conv_channels1, conv_channels2)
            self.conv3 = GatedConv(conv_channels2, conv_channels3)
            if large:
                self.conv4 = GatedConv(conv_channels3, conv_channels4)

        # Default convolution
        else:
            self.conv1 = Conv(features, conv_channels1)
            self.conv2 = Conv(conv_channels1, conv_channels2)
            self.conv3 = Conv(conv_channels2, conv_channels3)
            if large:
                self.conv4 = Conv(conv_channels3, conv_channels4)

        # Recurrent layer
        num_hidden = conv_channels_out + 11 if large else conv_channels_out + 5
        self.rnn = BiRNN(conv_channels_out, num_hidden, large=large, lstm=lstm, fcl=fcl, bidir=bidir)

    def forward(self, x):
        # (batch, frames, features)
        x = x.permute(0, 2, 1)

        # (batch, features/channels, frames)
        x = self.conv1(x)

        if self.residual_connections:
            x, r = self.conv2(x)
            x, r = self.conv3(x, r)
            if self.large:
                x, r = self.conv4(x, r)
            x = x * r
        else:
            x = self.conv2(x)
            x = self.conv3(x)
            if self.large:
                x = self.conv4(x)

        #   (batch, channels, frames)
        # ->(batch, frames, channels)
        x = self.rnn(x)

        # (batch, 2)
        return F.softmax(x, dim=1)


class DenseSingle(nn.Module):

    def __init__(self, input_size, output_size, dropout, dilation, padding, kernel_size, stride):
        super(DenseSingle, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=padding,
                      stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm1d(output_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        out = self.layer(x)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):

    def __init__(self, input_size, n_layers, growth_rate, dropout, dilation, padding, kernel_size, stride):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(DenseSingle(input_size + i * growth_rate, growth_rate,
                                      dropout, dilation, padding, kernel_size, stride))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(TransitionBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(output_size),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):
    def __init__(self, large=False):
        super(DenseNet, self).__init__()

        if large:

            dropout = 0.4

            self.cnn_in = nn.Sequential(
                nn.Conv1d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0,
                          dilation=4, bias=False),
                nn.BatchNorm1d(48),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.dense1 = DenseBlock(input_size=48, n_layers=8, growth_rate=4, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.trans1 = TransitionBlock(input_size=80, output_size=48, dropout=dropout)

            self.dense2 = DenseBlock(input_size=48, n_layers=8, growth_rate=4, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.cnn_out = nn.Sequential(
                nn.Conv1d(in_channels=80, out_channels=80, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(80),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.out = nn.Linear(80, 2, bias=False)

        else:

            dropout = 0.4

            self.cnn_in = nn.Sequential(
                nn.Conv1d(in_channels=24, out_channels=24, kernel_size=6, stride=1, padding=0,
                          dilation=4, bias=False),
                nn.BatchNorm1d(24),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.dense1 = DenseBlock(input_size=24, n_layers=6, growth_rate=3, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.trans1 = TransitionBlock(input_size=42, output_size=24, dropout=dropout)

            self.dense2 = DenseBlock(input_size=24, n_layers=6, growth_rate=3, kernel_size=3,
                                     dilation=1, padding=1, stride=1, dropout=dropout)

            self.cnn_out = nn.Sequential(
                nn.Conv1d(in_channels=42, out_channels=42, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(42),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            )

            self.out = nn.Linear(42, 2, bias=False)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.cnn_in(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.cnn_out(x)

        x = x.view(BATCH_SIZE, -1)

        return F.softmax(self.out(x), dim=1)
