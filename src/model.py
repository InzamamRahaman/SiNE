import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


def hadamard(x, y):
    return x * y


def average(x, y):
    return (x + y)/2.0


def l1(x, y):
    return np.abs(x - y)


def l2(x, y):
    return np.power(x - y, 2)


def concat(x, y):
    return np.concatenate((x, y), axis=1)


FEATURE_FUNCS = {
    'l1': l1,
    'l2': l2,
    'concat': concat,
    'average': average,
    'hadamard': hadamard
}


class SiNE(nn.Module):
    def __init__(self, num_nodes, dim1, dim2):
        super(SiNE, self).__init__()
        self.tanh = nn.Tanh()
        self.embeddings = nn.Embedding(num_nodes + 1, dim1)
        self.layer11 = nn.Linear(dim1, dim2, bias=False)
        self.layer12 = nn.Linear(dim1, dim2, bias=False)
        self.bias1 = Parameter(torch.zeros(1))
        self.layer2 = nn.Linear(dim2, 1, bias=False)
        self.bias2 = Parameter(torch.zeros(1))
        self.register_parameter('bias1', self.bias1)
        self.register_parameter('bias2', self.bias2)

    def forward(self, xi, xj, xk, delta):
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        z11 = self.tanh(self.layer11(i_emb) + self.layer12(j_emb) + self.bias1)
        z12 = self.tanh(self.layer11(i_emb) + self.layer12(k_emb) + self.bias1)

        f_pos = self.tanh(self.layer2(z11) + self.bias2)
        f_neg = self.tanh(self.layer2(z12) + self.bias2)

        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, f_pos + delta - f_neg)
        loss = torch.sum(loss)

        return loss

    def _regularizer(self, x):
        zeros = torch.zeros_like(x)
        normed = torch.norm(x - zeros, p=2)
        term = torch.pow(normed, 2)
        # print('The parameter of ', x)
        # print('Yields ',term)
        return term

    def regularize_weights(self):
        loss = 0
        for parameter in self.parameters():
            loss += self._regularizer(parameter)
        return loss

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([x]))
        emb = self.embeddings(x)
        emb = emb.data.numpy()[0]
        return emb

    def get_edge_feature(self, x, y, operation='hadamard'):
        func = FEATURE_FUNCS[operation]
        x = self.get_embedding(x)
        y = self.get_embedding(y)
        return func(x, y)




def tensorfy_col(x, col_idx):
    col = x[:,col_idx]
    col = torch.LongTensor(col)
    col = Variable(col)
    return col


def get_training_batch(triples, batch_size):
    nrows = triples.shape[0]
    rows = np.random.choice(nrows, batch_size, replace=False)
    choosen = triples[rows,:]
    xi = tensorfy_col(choosen, 0)
    xj = tensorfy_col(choosen, 1)
    xk = tensorfy_col(choosen, 2)
    return xi, xj, xk


def fit_model(sine, triplets, delta, batch_size, epochs, alpha,
                lr=0.4, weight_decay=0.0, print_loss=True):
    optimizer = optim.Adagrad(sine.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        sine.zero_grad()
        xi, xj, xk = get_training_batch(triplets, batch_size)
        loss = sine(xi, xj, xk, delta)
        # print(loss)
        regularizer_loss = alpha * sine.regularize_weights()
        # print(regularizer_loss)
        loss += regularizer_loss
        loss.backward()
        optimizer.step()
        if print_loss:
            print('Loss at epoch ', epoch + 1, ' is ', loss.data[0])
    return sine



