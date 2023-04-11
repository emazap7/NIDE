import time
import numpy as np
import logging
logger = logging.getLogger("idesolver")
logger.setLevel(logging.WARNING)#(logging.DEBUG)

#Torch libraries 
import torch 
from torch import Tensor
from torch import nn

#Custom libraries
import source.kernels as kernels
from source.solver import IDESolver_monoidal
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from torchdiffeq import odeint
from source.integrators import MonteCarlo
flatten_ode_parameters = kernels.flatten_ode_parameters
mc = MonteCarlo()
number_MC_samplings = 1000

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

class IDEF(nn.Module):
    
    def forward_with_grad(self, z_i, t_i, z, t, n_params,
                          grad_outputs, kernel, ode_func,
                          lower_bound, upper_bound,
                          n_ode_params, kernel_type_nn, ode_nn):
        
        batch_size = z_i.shape[0]
        
        n_dim = z_i.shape[2]
        
        
            #####################
        def _interpolate_y(y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:

            x=t
            y = y
            coeffs = natural_cubic_spline_coeffs(x, y)
            interpolation = NaturalCubicSpline(coeffs)

            def output(point:torch.Tensor):
                return interpolation.evaluate(point.to(device))

            return output
            ######################
        
        interpolated_z = _interpolate_y(z.reshape(z.size(0), z.size(2)))
        
        def adkFdz(x,s):
            
            mc = MonteCarlo()
            x = x.detach().requires_grad_(True)
            if kernel_type_nn is not True:
                transposed_kernel = torch.transpose(kernel(x,s),1,2)
            ######################
        
            z_int = interpolated_z(s)
            
            z_int = z_int.detach().requires_grad_(True)
            if kernel_type_nn is not True:
                out = torch.bmm(self.forward(z_int,t_i),transposed_kernel)
            else:  
                out = kernel.forward(self.forward(z_int,t_i).squeeze(1),x.repeat(s.size(0)).reshape(s.size(0),1),s).unsqueeze(1)

            a = torch.cat([grad_outputs]*number_MC_samplings)
            adfdt, *adfdp = torch.autograd.grad(
                (out,),(x,) + tuple(self.parameters()) + tuple(kernel.parameters()), 
                grad_outputs=(a),
                allow_unused=True, retain_graph=True
            )
            
            if adfdp is not None:
                adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
                adfdp = adfdp.expand(batch_size, -1) / batch_size
                adfdp = torch.stack([adfdp]*number_MC_samplings)
            
            #if adfdt is not None:
            #    adfdt = adfdt.expand(batch_size, 1) / batch_size
            #    adfdt = torch.stack([adfdt]*number_MC_samplings)
            #else: 
            adfdt = torch.zeros(number_MC_samplings,batch_size,1).to(device)   ###This is if time gradients are needed (cf. eq. 52 of NODE paper). Set to zero for now
            
            return torch.cat([out, adfdt, adfdp],-1)
            
        def integral(x):
            number_MC_samplings = 1000
            x = x.to(device)

            def integrand(s):

                s = s.to(device)
                
                return adkFdz(x,s)

            ####
            if lower_bound(x) < upper_bound(x):
                interval = [[lower_bound(x),upper_bound(x)]]
            else: 
                interval = [[upper_bound(x),lower_bound(x)]]
            ####
            '''
            return [torch.Tensor([mc.integrate(
                          fn= lambda s: torch.sign(upper_bound(x)-lower_bound(x)).to(device)*integrand(s)[:,:,i],
                           dim= 1,
                           N=number_MC_samplings,
                           integration_domain = interval #[[self.lower_bound(x),self.upper_bound(x)]] 
                           )]) for i in range (2*n_dim,2*n_dim+n_params+1)]#(2*n_dim+n_params+1)]
            '''
            
            return mc.integrate(
                          fn= lambda s: torch.sign(upper_bound(x)-lower_bound(x)).to(device)*integrand(s)[:,:,n_dim:n_dim+n_params-n_ode_params+1],
                           dim= 1,
                           N=number_MC_samplings,
                           integration_domain = interval, #[[self.lower_bound(x),self.upper_bound(x)]] 
                           out_dim = 0
                           )
        
        a_int_dfdt = integral(t_i)[:,0].to(device) #(this is needed if time gradients are wanted) + torch.matmul(grad_outputs,torch.matmul(kernel(t_i,torch.Tensor([t_i])),torch.transpose(self.forward(z_i,t_i),1,2)))[0,0,:].to(device)
        #a_int_dfdt = integral(t_i)[2*n_dim].to(device)
        
        a_int_dfdp = integral(t_i)[:,1:].to(device)#torch.cat(integral(t_i)[1:]).to(device)
        #a_int_dfdp = torch.cat(integral(t_i)[2*n_dim+1:]).to(device)
        
        #a_int_dfdz = torch.cat(integral(t_i)[n_dim:2*n_dim]).to(device)
        z_i = z_i.detach().requires_grad_(True)
        t_i = t_i.detach().requires_grad_(True)
        
        func_i = self.forward(z_i, t_i)
        if kernel_type_nn is not True:
            functional_der = torch.matmul(func_i,torch.transpose(kernel(t_i,torch.Tensor([t_i]).to(device)),1,2))
        else:
            functional_der = kernel.forward(func_i.squeeze(1),t_i.reshape(1,1),t_i.reshape(1,1)).unsqueeze(1)
            
        adfdz = torch.autograd.grad(
        (functional_der,), (z_i), grad_outputs = (grad_outputs),
         allow_unused=True, retain_graph=True
        )
        
        if ode_nn is True:
            ode_out = ode_func.forward(t_i,z_i)
            adfdz_ode = torch.autograd.grad(
        (ode_out.view(z_i.size()),), (z_i), grad_outputs = (grad_outputs),
         allow_unused=True, retain_graph=True
        )
            
            adfdp_ode = torch.autograd.grad(
        (ode_out.view(z_i.size()),), tuple(ode_func.parameters()), grad_outputs = (grad_outputs),
         allow_unused=True, retain_graph=True
        )
            
            if adfdz_ode[0] is not None:
                adfdz = torch.cat(adfdz) + torch.cat(adfdz_ode)
            else:
                adfdz = torch.cat(adfdz)
            
            adfdp_ode = torch.cat([p_grad.flatten() for p_grad in adfdp_ode]).unsqueeze(0)
               
            All_tensors = ode_out+integral(t_i)[:,:n_dim].to(device), adfdz, torch.cat([a_int_dfdp,adfdp_ode],-1), a_int_dfdt
        else:
            adfdz = torch.cat(adfdz)
        
            All_tensors = integral(t_i)[:,:n_dim].to(device), adfdz, a_int_dfdp, a_int_dfdt #torch.cat(integral(t_i)[:n_dim]).to(device), adfdz, a_int_dfdp, a_int_dfdt
        
        return All_tensors
    
    

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
        
  
      
class IDEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, n_ode_params, 
                func, ker, ode_func,
                lower_bound_function,
                upper_bound_function,
                kernel_type_nn, ode_nn):
        start = time.time()
        #assert isinstance(func, IDEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)
        
        # print('[forward] z0.shape: ',z0.shape)

        with torch.no_grad():
            z = torch.zeros(time_len,bs,*z_shape).to(z0)
            z[0] = z0
            '''
            for i_t in range(time_len - 1):
                
                solver =  IDESolver_monoidal(x = torch.linspace(t[i_t],t[i_t+1],5).to(device),
                                    y_0 = z0.flatten().to(device), 
                                    c = (lambda x,y:ode_func.forward(x,y)) if ode_func is not None else None, 
                                    #d = lambda x: torch.Tensor([1]).to(device), 
                                    k = ker, 
                                    f = lambda y:func(y.to(device),t[i_t].to(device)).to(device),
                                    lower_bound = lower_bound_function,
                                    upper_bound = upper_bound_function,
                                    max_iterations = 2,
                                    kernel_nn = kernel_type_nn,
                                    ode_atol = 1e-3,
                                    ode_rtol= 1e-3,
                                    int_atol = 1e-3,
                                    int_rtol = 1e-3,
                                    integration_dim = 0)
            
                solver.solve()

                z0 = solver.y[solver.y.size(0)-1,:]
                   
                z[i_t+1] = z0
                '''
            solver = IDESolver_monoidal(x = t.to(device),
                                    y_0 = z0.flatten().to(device), 
                                    c = (lambda x,y:ode_func.forward(x,y)) if ode_func is not None else None, 
                                    #d = lambda x: torch.Tensor([1]).to(device), 
                                    k = ker, 
                                    f = lambda y:func(y.to(device),t[0].to(device)).to(device),
                                    lower_bound = lower_bound_function,
                                    upper_bound = upper_bound_function,
                                    max_iterations = 3,
                                    kernel_nn = kernel_type_nn,
                                    ode_atol = 1e-4,
                                    ode_rtol= 1e-4,
                                    int_atol = 1e-4,
                                    int_rtol = 1e-4,
                                    integration_dim = 0)
            
            solver.solve()

            z = solver.y
            z = z.unsqueeze(1)
                   
            
        ctx.func = func
        ctx.ode_func = ode_func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        ctx.kernel = ker
        ctx.lower_bound = lower_bound_function
        ctx.upper_bound = upper_bound_function
        ctx.n_ode_params = n_ode_params
        ctx.kernel_type_nn = kernel_type_nn
        ctx.ode_nn = ode_nn
        
        end = time.time()
        # print(f"Training time: {(end-start):.3f} minutes")
        
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        
        func = ctx.func
        kernel = ctx.kernel
        ode_func = ctx.ode_func
        lower_bound = ctx.lower_bound
        upper_bound = ctx.upper_bound
        n_ode_params = ctx.n_ode_params
        kernel_type_nn = ctx.kernel_type_nn
        ode_nn = ctx.ode_nn
        
        t, z, flat_parameters = ctx.saved_tensors
        # print('[backward] z.shape: ',z.shape)
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0) 
           
        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(t_i, aug_z_i):
           
            z_i, a = aug_z_i[:,:n_dim], aug_z_i[:,n_dim:2*n_dim]  

            # Unflatten z and a  
            z_i = z_i.unsqueeze(1)
            a = a.unsqueeze(1)
            

            with torch.set_grad_enabled(True):
                
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)    
                
                func_eval, adfdz, adfdp, adfdt = func.forward_with_grad(z_i, t_i, z, t, 
                                                                        n_params, a, kernel,
                                                                        ode_func, lower_bound, 
                                                                        upper_bound, n_ode_params,
                                                                        kernel_type_nn, ode_nn)  
                
                adfdp = adfdp#.unsqueeze(1)
               
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            adfdp = adfdp.view(bs,n_params)
            adfdt = adfdt.view(bs,1)
            
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=-1)
        
        
        
        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz 
    
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            #  Return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)
            
            
            ########
            f_i = torch.zeros(bs,n_dim)
            
            #######
            
            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                ################################### Need to integrate the func
                #f_i = func(z_i, t_i).view(bs, n_dim)
                ###################################
                #'''
                def _interpolate_y(y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:
                    x=t
                    y = y
                    coeffs = natural_cubic_spline_coeffs(x, y)
                    interpolation = NaturalCubicSpline(coeffs)

                    def output(point:torch.Tensor):
                        return interpolation.evaluate(point.to(device))

                    return output
                    ######################
                interpolated_z = _interpolate_y(z.reshape(z.size(0), z.size(2)))
                '''
                f_i_list = [torch.Tensor([mc.integrate(
                                fn = lambda s: torch.bmm(func(interpolated_z(s),t_i),torch.transpose(cos_kernel(t_i,s),1,2))[:,:,i],
                                dim= 1,
                                N=number_MC_samplings,
                                integration_domain = [[torch.Tensor([0]),torch.Tensor([t_i])]]
                                )]) for i in range(n_dim)]
                
                
                
                print('func(interpolated_z(s),t_i).shape: ',func(interpolated_z(s),t_i).shape)
                print('torch.transpose(kernel(t_i,s),1,2).shape: ',torch.transpose(kernel(t_i,s),1,2).shape)
                f_i_list = mc.integrate(
                                fn = lambda s: torch.bmm(func(interpolated_z(s),t_i),torch.transpose(kernel(t_i,s),1,2))[:,:,:n_dim],
                                dim= 1,
                                N=number_MC_samplings,
                                integration_domain = [[torch.Tensor([0]),torch.Tensor([t_i])]],
                                out_dim = 0
                                )
                f_i = f_i_list#torch.cat(f_i_list).unsqueeze(0)
                '''

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.zeros(1,1).to(device)#torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i


                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)
                # print('[backward] aug_z.shape: ',aug_z.shape)

                
                # Solve augmented system backwards
                n_aug_z = aug_z.size()
                
####################################################################
                aug_ans = odeint(augmented_dynamics, aug_z,torch.linspace(t[i_t],t[i_t-1],4).to(device),atol = 1e-4,rtol = 1e-4,method='dopri5')#,options=dict(step_size=1e-5))
                # print('[backward] aug_ans.shape: ',aug_ans.shape)
####################################################################                
                
                # Unpack solved backwards augmented system
                
                adj_z[:] = aug_ans[aug_ans.size(0)-1,:,n_dim:2*n_dim]
                adj_p[:] += aug_ans[aug_ans.size(0)-1,:,2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[aug_ans.size(0)-1,:,2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.zeros(1,1).to(device)#torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]


            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None, None, None, None, None, None, None, None
        
        
class NeuralIDE(nn.Module):
    def __init__(self, func, kernel, ode_func, 
                 lower_bound_function = lambda x: torch.Tensor([0]).to(device),
                 upper_bound_function = lambda x: x,
                 kernel_type_nn = True, ode_nn = False):
        super(NeuralIDE, self).__init__()
        #assert isinstance(func, IDEF)
        self.func = func
        self.kernel = kernel
        self.ode_func = ode_func
        self.lower_bound_function = lower_bound_function
        self.upper_bound_function = upper_bound_function
        self.kernel_type_nn = kernel_type_nn
        self.ode_nn = ode_nn

    def forward(self, z0, t=Tensor([0., 1.]),return_whole_sequence=False):
        t = t.to(z0)
        
        if self.kernel_type_nn is True:
            parameters = torch.cat([self.func.flatten_parameters(),kernels.flatten_kernel_parameters(self.kernel)])
        else:
            parameters = self.func.flatten_parameters()
        
        if self.ode_nn is True:
            parameters = torch.cat([parameters,flatten_ode_parameters(self.ode_func)])
            n_ode_params = torch.cat([flatten_ode_parameters(self.ode_func)]).size(0)
        else:
            n_ode_params = 0
        
        z = IDEAdjoint.apply(z0, t, parameters, n_ode_params, 
                             self.func, self.kernel, self.ode_func,
                            self.lower_bound_function, self.upper_bound_function,
                            self.kernel_type_nn,self.ode_nn)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
            


class LinearIDEF(IDEF):
    def __init__(self, W):
        super(LinearIDEF, self).__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.lin.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.lin(x)

class SpiralFunctionExample(LinearIDEF):
    def __init__(self):
        super(SpiralFunctionExample, self).__init__(Tensor([[0.31,  0.71],[ -0.57, 0.41]]).to(device))


T = torch.randn(2, 2)/4

class RandomLinearIDEF(LinearIDEF):
    def __init__(self):
        super(RandomLinearIDEF, self).__init__(T)
        
def to_np(x):
    return x.detach().cpu().numpy()



class NNIDEF(IDEF):
    def __init__(self, in_dim,out_dim,shapes,NL=nn.ELU):
        super(NNIDEF, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t):
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))
        y = self.last.forward(y)

        return y
    
class Simple_NN(nn.Module):
    def __init__(self, in_dim, hid_dim,out_dim):
        super(Simple_NN, self).__init__()

        self.lin1 = nn.Linear(in_dim+1, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, hid_dim)
        self.lin4 = nn.Linear(hid_dim, out_dim)
        self.ELU = nn.ELU(inplace=True)
        
        self.in_dim = in_dim

    def forward(self,x,y):
        y = y.reshape(self.in_dim).to(device)
        x = x.reshape(1).to(device)
        
        y_in = torch.cat([x,y],-1)
        h = self.ELU(self.lin1(y_in))
        h = self.ELU(self.lin2(h))
        h = self.ELU(self.lin3(h))
        out = self.lin4(h)
        
        return out
    
    
class IDEF_wODE(nn.Module):
    
    def forward_with_grad(self, z_i, t_i, z, t, n_params,
                          grad_outputs, kernel, ode_func,
                          lower_bound, upper_bound,
                          n_ode_params, kernel_type_nn, ode_nn,ode_mode):
        
        batch_size = z_i.shape[0]
        
        n_dim = z_i.shape[2]
        
        if ode_mode is not True:
            #####################
            def _interpolate_y(y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:

                x=t
                y = y
                coeffs = natural_cubic_spline_coeffs(x, y)
                interpolation = NaturalCubicSpline(coeffs)

                def output(point:torch.Tensor):
                    return interpolation.evaluate(point.to(device))

                return output
                ######################

            interpolated_z = _interpolate_y(z.reshape(z.size(0), z.size(2)))

            def adkFdz(x,s):

                mc = MonteCarlo()
                x = x.detach().requires_grad_(True)
                if kernel_type_nn is not True:
                    transposed_kernel = torch.transpose(kernel(x,s),1,2)
                ######################

                z_int = interpolated_z(s)

                z_int = z_int.detach().requires_grad_(True)
                if kernel_type_nn is not True:
                    out = torch.bmm(self.forward(z_int,t_i),transposed_kernel)
                else:  
                    out = kernel.forward(self.forward(z_int,t_i).squeeze(1),x.repeat(s.size(0)).reshape(s.size(0),1),s).unsqueeze(1)

                a = torch.cat([grad_outputs]*number_MC_samplings)
                adfdz, adfdt, *adfdp = torch.autograd.grad(
                    (out,),(z_int,x) + tuple(self.parameters()) + tuple(kernel.parameters()), 
                    grad_outputs=(a),
                    allow_unused=True, retain_graph=True
                )

                if adfdp is not None:
                    adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
                    adfdp = adfdp.expand(batch_size, -1) / batch_size
                    adfdp = torch.stack([adfdp]*number_MC_samplings)

                 
                adfdt = torch.zeros(number_MC_samplings,batch_size,1).to(device)

                return torch.cat([out, adfdz, adfdt, adfdp],-1)

            def integral(x):
                number_MC_samplings = 1000
                x = x.to(device)

                def integrand(s):

                    s = s.to(device)

                    return adkFdz(x,s)

                ####
                if lower_bound(x) < upper_bound(x):
                    interval = [[lower_bound(x),upper_bound(x)]]
                else: 
                    interval = [[upper_bound(x),lower_bound(x)]]
                ####
                

                return mc.integrate(
                              fn= lambda s: torch.sign(upper_bound(x)-lower_bound(x)).to(device)*integrand(s)[:,:,2*n_dim:2*n_dim+n_params-n_ode_params+1],
                               dim= 1,
                               N=number_MC_samplings,
                               integration_domain = interval, 
                               out_dim = 0
                               )

            a_int_dfdt = integral(t_i)[:,0].to(device) 

            a_int_dfdp = integral(t_i)[:,1:].to(device)
            

            
            z_i = z_i.detach().requires_grad_(True)
            t_i = t_i.detach().requires_grad_(True)
            
            func_i = self.forward(z_i, t_i)
            
            if kernel_type_nn is not True:
                functional_der = torch.matmul(func_i,torch.transpose(kernel(t_i,torch.Tensor([t_i]).to(device)),1,2))
            else:
                functional_der = kernel.forward(func_i.squeeze(1),t_i.reshape(1,1),t_i.reshape(1,1)).unsqueeze(1)

            adfdz = torch.autograd.grad(
            (functional_der,), (z_i), grad_outputs = (grad_outputs),
             allow_unused=True, retain_graph=True
            )

            if ode_nn is True:
                ode_out = ode_func.forward(t_i,z_i)
                adfdz_ode = torch.autograd.grad(
            (ode_out.view(z_i.size()),), (z_i), grad_outputs = (grad_outputs),
             allow_unused=True, retain_graph=True
            )

                adfdp_ode = torch.autograd.grad(
            (ode_out.view(z_i.size()),), tuple(ode_func.parameters()), grad_outputs = (grad_outputs),
             allow_unused=True, retain_graph=True
            )

                if adfdz_ode[0] is not None:
                    adfdz = torch.cat(adfdz) + torch.cat(adfdz_ode)
                else:
                    adfdz = torch.cat(adfdz)

                adfdp_ode = torch.cat([p_grad.flatten() for p_grad in adfdp_ode]).unsqueeze(0)

                All_tensors = ode_out+integral(t_i)[:,:n_dim].to(device), adfdz, torch.cat([a_int_dfdp,adfdp_ode],-1), a_int_dfdt
            else:
                adfdz = torch.cat(adfdz)

                All_tensors = integral(t_i)[:,:n_dim].to(device), adfdz, a_int_dfdp, a_int_dfdt 
        
        
        else: 
            ode_out = ode_func.forward(t_i,z_i)
            adfdz_ode = torch.autograd.grad(
            (ode_out.view(z_i.size()),), (z_i), grad_outputs = (grad_outputs),
             allow_unused=True, retain_graph=True
            )

            adfdp_ode = torch.autograd.grad(
            (ode_out.view(z_i.size()),), tuple(ode_func.parameters()), grad_outputs = (grad_outputs),
             allow_unused=True, retain_graph=True
            )
            
            adfdz_ode = torch.cat(adfdz_ode)
            adfdp_ode = torch.cat([p_grad.flatten() for p_grad in adfdp_ode]).unsqueeze(0)
                
            All_tensors = ode_out, adfdz_ode, adfdp_ode, torch.zeros(1,1).to(device)
            
        return All_tensors
    
    

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
        

class IDEAdjoint_wODE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, n_ode_params, 
                func, ker, ode_func,
                lower_bound_function,
                upper_bound_function,
                kernel_type_nn, ode_nn, ode_mode, args):
        
        bs, *z_shape = z0.size()
        time_len = t.size(0)
        

        with torch.no_grad():
            z = torch.zeros(time_len,bs,*z_shape).to(z0)
            z[0] = z0

            solver = IDESolver_monoidal(x = t.to(device),
                                    y_0 = z0.flatten().to(device), 
                                    c = (lambda x,y:ode_func.forward(x,y)) if ode_func is not None else None, 
                                    #d = lambda x: torch.Tensor([1]).to(device), 
                                    k = ker, 
                                    f = lambda y:func(y.to(device),t[0].to(device)).to(device),
                                    lower_bound = lower_bound_function,
                                    upper_bound = upper_bound_function,
                                    ode_option = ode_mode,
                                    max_iterations = args.max_iterations,
                                    kernel_nn = kernel_type_nn,
                                    ode_atol = args.ode_atol,
                                    ode_rtol= args.ode_rtol,
                                    int_atol = args.int_atol,
                                    int_rtol = args.int_rtol,
                                    integration_dim = 0)
            
            solver.solve()

            z = solver.y
            z = z.unsqueeze(1)
                   
            
        ctx.func = func
        ctx.ode_func = ode_func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        ctx.kernel = ker
        ctx.lower_bound = lower_bound_function
        ctx.upper_bound = upper_bound_function
        ctx.n_ode_params = n_ode_params
        ctx.kernel_type_nn = kernel_type_nn
        ctx.ode_nn = ode_nn
        ctx.ode_mode = ode_mode
        ctx.args = args
        
        
        return z

    @staticmethod
    def backward(ctx, dLdz):
        
        
        func = ctx.func
        kernel = ctx.kernel
        ode_func = ctx.ode_func
        lower_bound = ctx.lower_bound
        upper_bound = ctx.upper_bound
        n_ode_params = ctx.n_ode_params
        kernel_type_nn = ctx.kernel_type_nn
        ode_nn = ctx.ode_nn
        ode_mode = ctx.ode_mode
        args = ctx.args
        
        t, z, flat_parameters = ctx.saved_tensors
        # print('[backward] z.shape: ',z.shape)
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0) 
           
        
        def augmented_dynamics(t_i, aug_z_i):
           
            z_i, a = aug_z_i[:,:n_dim], aug_z_i[:,n_dim:2*n_dim]  

              
            z_i = z_i.unsqueeze(1)
            a = a.unsqueeze(1)
            

            with torch.set_grad_enabled(True):
                
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)    
                
                func_eval, adfdz, adfdp, adfdt = func.forward_with_grad(z_i, t_i, z, t, 
                                                                        n_params, a, kernel,
                                                                        ode_func, lower_bound, 
                                                                        upper_bound, n_ode_params,
                                                                        kernel_type_nn, ode_nn, ode_mode)  
                
                adfdp = adfdp
               
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            adfdp = adfdp.view(bs,n_params)
            adfdt = adfdt.view(bs,1)
            
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=-1)
        
        
        
        dLdz = dLdz.view(time_len, bs, n_dim)  
    
        with torch.no_grad():
            
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)
            
            
            ########
            f_i = torch.zeros(bs,n_dim)
            
            #######
            
            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                
                def _interpolate_y(y: torch.Tensor):
                    x=t
                    y = y
                    coeffs = natural_cubic_spline_coeffs(x, y)
                    interpolation = NaturalCubicSpline(coeffs)

                    def output(point:torch.Tensor):
                        return interpolation.evaluate(point.to(device))

                    return output
                    ######################
                interpolated_z = _interpolate_y(z.reshape(z.size(0), z.size(2)))
                

                
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.zeros(1,1).to(device)

                
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i


                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)
                

                n_aug_z = aug_z.size()
                
####################################################################
                aug_ans = odeint(augmented_dynamics, aug_z,torch.linspace(t[i_t],t[i_t-1],4).to(device),atol = args.ode_atol,rtol = args.ode_rtol,method='dopri5')
####################################################################                
                
                # Unpack solved backwards augmented system
                
                adj_z[:] = aug_ans[aug_ans.size(0)-1,:,n_dim:2*n_dim]
                adj_p[:] += aug_ans[aug_ans.size(0)-1,:,2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[aug_ans.size(0)-1,:,2*n_dim + n_params:]

                del aug_z, aug_ans

            
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.zeros(1,1).to(device)


            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None, None, None, None, None, None, None, None, None, None
        

class NNIDEF_wODE(IDEF_wODE):
    def __init__(self, in_dim,out_dim,shapes,NL=nn.ELU):
        super(NNIDEF_wODE, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t):
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))
        y = self.last.forward(y)

        return y
    
    
class NeuralIDE_wODE(nn.Module):
    def __init__(self, func, kernel, ode_func, 
                 lower_bound_function = lambda x: torch.Tensor([0]).to(device),
                 upper_bound_function = lambda x: x,
                 kernel_type_nn = True, ode_nn = False, ode_mode = False, args=None):
        super(NeuralIDE_wODE, self).__init__()
        
        self.args = args
        global device
        device = args.device

        self.func = func
        self.kernel = kernel
        self.ode_func = ode_func
        self.lower_bound_function = lower_bound_function
        self.upper_bound_function = upper_bound_function
        self.kernel_type_nn = kernel_type_nn
        self.ode_nn = ode_nn
        self.ode_mode = ode_mode


    def forward(self, z0, t=Tensor([0., 1.]),return_whole_sequence=False):
        t = t.to(z0)
        
        if self.ode_mode is not True:
        
            if self.kernel_type_nn is True:
                parameters = torch.cat([self.func.flatten_parameters(),kernels.flatten_kernel_parameters(self.kernel)])
            else:
                parameters = self.func.flatten_parameters()

            if self.ode_nn is True:
                parameters = torch.cat([parameters,flatten_ode_parameters(self.ode_func)])
                n_ode_params = torch.cat([flatten_ode_parameters(self.ode_func)]).size(0)
            else:
                n_ode_params = 0
        
        else:
            parameters = torch.cat([flatten_ode_parameters(self.ode_func)])
            n_ode_params = torch.cat([flatten_ode_parameters(self.ode_func)]).size(0)
        
        z = IDEAdjoint_wODE.apply(z0, t, parameters, n_ode_params, 
                             self.func, self.kernel, self.ode_func,
                            self.lower_bound_function, self.upper_bound_function,
                            self.kernel_type_nn,self.ode_nn,self.ode_mode,self.args)
        if return_whole_sequence:
            return z
        else:
            return z[-1]
 
