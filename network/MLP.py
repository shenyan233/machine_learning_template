import torch.nn as nn
from network_module.activation import jdlu, JDLU


class MLPLayer(nn.Module):
    def __init__(self, dim_in, dim_out, res_coef=0.0, dropout_p=0.1):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.res_coef = res_coef
        self.activation = nn.ReLU()
        self.activation1 = JDLU(dim_out)
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(dim_out)

    def forward(self, x):
        y = self.linear(x)
        y = self.activation1(y)
        # y = jdlu(y)
        y = self.dropout(y)
        if self.res_coef == 0:
            return self.ln(y)
        else:
            return self.ln(self.res_coef * x + y)


class MLP(nn.Module):
    def __init__(self, dim_in, dim, res_coef=0.5, dropout_p=0.1, n_layers=10):
        super().__init__()
        self.mlp = nn.ModuleList()
        self.first_linear = MLPLayer(dim_in, dim)
        self.n_layers = n_layers
        for i in range(n_layers):
            self.mlp.append(MLPLayer(dim, dim, res_coef, dropout_p))
        self.final = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.first_linear(x)
        for layer in self.mlp:
            x = layer(x)
        x = self.final(x)
        return x.squeeze()
