from torch import nn
import torch

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)

def linear_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat((x-s.reshape(s.size()[0],1,1),x+1-s.reshape(s.size()[0],1,1)),-1)
    B = torch.cat((x+1-s.reshape(s.size()[0],1,1),x-s.reshape(s.size()[0],1,1)),-1)
    C=torch.stack((A,B),-2)
    return C.reshape(s.size(0),2,2)  

def exp_kernel(x,s):
    A = torch.Tensor([torch.exp(-x),torch.tensor([0])])
    B = torch.Tensor([torch.tensor([0]),torch.exp(-x)])
    C=torch.stack((A,B),-2)
    return C.repeat(s.size(0),1,1)

def exp_kernel2(x,s):
    A = torch.cat([torch.exp(x/(1+s)).reshape(s.size(0),1),torch.zeros(s.size(0),1)],-1).to(device)
    B = torch.cat([torch.zeros(s.size(0),1),torch.exp(x/(1+s)).reshape(s.size(0),1)],-1).to(device)
    C = torch.stack([A,B],-1).to(device)
    return C

def exp_kernel3(x,s):
    A = torch.cat([torch.exp(x-s).reshape(s.size(0),1),torch.zeros(s.size(0),1)],-1).to(device)
    B = torch.cat([torch.zeros(s.size(0),1),torch.exp(x-s).reshape(s.size(0),1)],-1).to(device)
    C = torch.stack([A,B],-1).to(device)
    return C

def identity_kernel(x,s,dim):
    A = torch.ones(dim).to(device)
    B = torch.diag(A)
    return B.repeat(s.size(0),1,1)

def diagonal_cubic(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat((x**3-(s**3).reshape(s.size(0),1,1),torch.ones(s.size(0),1,1).to(device)),-1)
    B = torch.cat((torch.ones(s.size(0),1,1).to(device),x**3-(s**3).reshape(s.size(0),1,1)),-1)
    C=torch.stack((A,B),-2)
    return C.reshape(s.size(0),2,2)  

def cos_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.cos(x-s).reshape(s.size(0),1),-torch.sin(x-s).reshape(s.size(0),1)],-1)
    B = torch.cat([-torch.sin(x-s).reshape(s.size(0),1),-torch.cos(x-s).reshape(s.size(0),1)],-1)
    C = torch.stack([A,B],-1)
    return C

def tanh_kernel(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.tanh(x-s).reshape(s.size(0),1),1-torch.sinh(x-s).reshape(s.size(0),1)],-1)
    B = torch.cat([1-torch.cosh(x-s).reshape(s.size(0),1),-torch.tanh(x-s).reshape(s.size(0),1)],-1)
    C = torch.stack([A,B],-1)
    return C

def tanh_kernel_4D(x,s):
    x = x.to(device)
    s = s.to(device)
    A = torch.cat([torch.tanh(x-s).reshape(s.size(0),1),1-torch.sinh(x-s).reshape(s.size(0),1),torch.ones(s.size(0),1).to(device),torch.zeros(s.size(0),1).to(device)],-1)
    B = torch.cat([1-torch.cosh(x-s).reshape(s.size(0),1),-torch.tanh(x-s).reshape(s.size(0),1),1-torch.cosh(x-s).reshape(s.size(0),1),torch.tanh(x-s).reshape(s.size(0),1)],-1)
    C = torch.cat([1+torch.cos(s).reshape(s.size(0),1),-torch.tanh(x-s).reshape(s.size(0),1),torch.sinh(x-s).reshape(s.size(0),1),torch.tanh(x-s).reshape(s.size(0),1)],-1)
    D = torch.cat([torch.cosh(x-s).reshape(s.size(0),1),5-torch.sinh(x-s).reshape(s.size(0),1),3+torch.sinh(x-s).reshape(s.size(0),1),-torch.tanh(x-s).reshape(s.size(0),1)],-1)
    F = torch.stack([A,B,C,D],-1)
    return F


class neural_kernel(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        '''
        Give a tuple with several hidden dim options (future work)
        '''
        
        super(neural_kernel, self).__init__()

        self.lin1 = nn.Linear(in_dim+2, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim,hid_dim)
        self.lin4 = nn.Linear(hid_dim, hid_dim)
        self.lin5 = nn.Linear(hid_dim, hid_dim)
        self.lin6 = nn.Linear(hid_dim, hid_dim)
        self.lin7 = nn.Linear(hid_dim, hid_dim)
        self.lin8 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)

    def forward(self,y, x, t):
        y_in = torch.cat([y,x,t],-1)
        h = self.ELU(self.lin1(y_in))
        h = self.ELU(self.lin2(h))
        h = self.ELU(self.lin3(h))
        h = self.ELU(self.lin4(h))
        h = self.ELU(self.lin5(h))
        h = self.ELU(self.lin6(h))
        h = self.ELU(self.lin7(h))
        out = self.lin8(h)
        
        return out


class kernel_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(kernel_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes)-1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))
        y = self.last.forward(y)

        return y
    
class kernel_NN_nbatch(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(kernel_NN_nbatch, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes)-1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        t = t.unsqueeze(0).repeat(y.shape[0],1,1)
        s = s.unsqueeze(0).repeat(y.shape[0],1,1)
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))
        y = self.last.forward(y)

        return y
    
class linear_nn(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linear_nn, self).__init__()
        
        self.lin = nn.Linear(in_dim,out_dim,1)
        
    def forward(self,x):
        return self.lin(x)
    
def flatten_kernel_parameters(kernel):
    p_shapes = []
    flat_parameters = []
    for p in kernel.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


def flatten_F_parameters(NN_F):
    p_shapes = []
    flat_parameters = []
    for p in NN_F.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)


def flatten_ode_parameters(ode_func):
    p_shapes = []
    flat_parameters = []
    for p in ode_func.parameters():
        p_shapes.append(p.size())
        flat_parameters.append(p.flatten())
    return torch.cat(flat_parameters)

