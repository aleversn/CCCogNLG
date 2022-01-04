import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def __init__(self, input_size):
        super(GCN, self).__init__()
        self.transform = nn.Linear(input_size, input_size, bias=False)
        self.diffusion = nn.Linear(input_size, input_size, bias=False)
        self.retained = nn.Linear(input_size, input_size, bias=False)
        self.predict = MLP(input_sizes=(input_size, input_size, 2))

        self.H_hop = MLP(input_sizes=(input_size * 2, input_size))
        self.apply(self.init_weights)
    
    def gelu(self, x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def get_hop_features(self, x):
        return self.H_hop(x)

    def forward(self, A, x, semantic_expand):
        # 输入的A = AD-1
        # 这里用的激活函数也是Gelu
        # ∆ = σ((AD^−1)^T σ(XW_1))
        # X^' = σ(XW_2 + ∆)
        x = self.transform(x.transpose(1, 2)).transpose(1, 2)
        layer1_diffusion = A.transpose(1, 2).matmul(self.gelu(self.diffusion(x)))
        x = self.gelu(self.retained(x) + layer1_diffusion)
        layer2_diffusion = A.transpose(1, 2).matmul(self.gelu(self.diffusion(x)))
        x = self.gelu(self.retained(x) + layer2_diffusion)
        H = self.get_hop_features(torch.cat((x, semantic_expand), dim=2))
        p = self.predict(H).squeeze(-1)
        return F.softmax(p, dim=-1)

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)
    
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = self.gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x