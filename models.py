import torch
import torch.nn as nn
from torch.autograd import Variable

class BaseModel(object):
    def __init__(self, device):
        self.device = device

    def forward_prop(self):
        raise NotImplementedError

    def optimization_params(self):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

class LinearModel(BaseModel):
    def __init__(self, in_dim, out_dim, device):
        super(LinearModel, self).__init__(device)
        self.linear_func = nn.Linear(in_dim, out_dim, bias=False).to(self.device)

    def forward_prop(self, X):
        X = Variable(torch.from_numpy(X), requires_grad=False).to(self.device)
        X_hat = self.linear_func(X)

        # Remove mean and normalize so each row is unit norm. By doing this, 
        # np.matmul(X, X.T) is just the correlation matrix.
        X_hat = X_hat - torch.mean(X_hat, 1, keepdim=True)
        X_hat = X_hat / torch.norm(X_hat, dim=1, keepdim=True)
        R_hat = torch.matmul(X_hat, torch.transpose(X_hat, 0, 1))

        return R_hat

    def optimization_params(self):
        return self.linear_func.parameters()

    def get_weights(self):
        return self.linear_func.weight.data.cpu().numpy()

class ElementWiseScalingModel(BaseModel):
    def __init__(self, dim, device):
        super(ElementWiseScalingModel, self).__init__(device)
        self.W = Variable(torch.rand(1, dim).to(self.device), requires_grad=True)
    
    def forward_prop(self, X):
        X = Variable(torch.from_numpy(X), requires_grad=False).to(self.device)

        X_hat = X * self.W
        X_hat = X_hat - torch.mean(X_hat, 1, keepdim=True)
        X_hat = X_hat / torch.norm(X_hat, dim=1, keepdim=True)
        R_hat = torch.matmul(X_hat, torch.transpose(X_hat, 0, 1))
        
        return R_hat
        
    def optimization_params(self):
        return [self.W]

    def get_weights(self):
        return self.W.data.cpu().numpy()

