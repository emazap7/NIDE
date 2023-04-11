import os
import shutil
import torch
import numpy as np
import pickle
# import yaml


# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
from torch.utils.data import Dataset
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from source.integrators import MonteCarlo
mc = MonteCarlo()

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_dict_template():
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "labels": None
            }

def normalize_data(data):
    reshaped = data.reshape(-1, data.size(-1))

    att_min = torch.min(reshaped, 0)[0]
    att_max = torch.max(reshaped, 0)[0]

    # we don't want to divide by zero
    att_max[ att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    return data_norm, att_min, att_max

def display_video(frames, framerate, filename=None):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
        im.set_data(frame)
        return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    
    if filename is not None: anim.save(filename)
    return HTML(anim.to_html5_video())


    
def get_system_definition(name, mode='rb'):
    with open(name, mode=mode) as f:
        return f.read()
    
    
class Train_val_split:
    def __init__(self, IDs,val_size_fraction):
        
        
        IDs = np.random.permutation(IDs)
        # print('IDs: ',IDs)
        self.IDs = IDs
        self.val_size = int(val_size_fraction*len(IDs))
    
    def train_IDs(self):
        train = sorted(self.IDs[:len(self.IDs)-self.val_size])
        # print('len(train): ',len(train))
        # print('train: ',train)
        return train
    
    def val_IDs(self):
        val = sorted(self.IDs[len(self.IDs)-self.val_size:])
        # print('len(val): ',len(val))
        # print('val: ',val)
        return val

        
# class Dynamics_Dataset(torch.utils.data.Dataset):
class Dynamics_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        # self.batch_size = batch_size

    def __getitem__(self, index):
        # print('index: ',index)
        # print('self.list_IDs.shape: ',len(self.list_IDs))
        # print('self.Data: ',self.Data)
        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        t = self.times#[ID]

        return obs, t, ID
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.Data)
    
    
    
class Dynamics_Dataset_Video(Dataset):
    def __init__(self, Data, times, range_segment):
        self.times = times.float()
        self.Data = Data.float()
        self.range_segment=range_segment
        

    def __getitem__(self, index):
        possible_IDs = torch.arange(index,index+self.range_segment)
        print('possible_IDs: ',possible_IDs)
        IDs = possible_IDs[torch.randint(len(possible_IDs))]
        print('IDs: ',IDs)
        
        
        obs = self.Data[IDs]
        t = self.times[IDs]
        
        print('IDs: ',IDs)
        print('t: ',t)

        return obs, t, ID

    def __len__(self):
        return len(self.times)
    

    
class Test_Dynamics_Dataset(torch.utils.data.Dataset):
    def __init__(self, Data, times):


        self.times = times.float()
        self.Data = Data.float()

    def __len__(self):
        return len(self.Data)

    def __getitem__(self,index):

        ID = index 
        obs = self.Data[ID]
        t = self.times

        return obs, t, ID

    
class LRScheduler():

    def __init__(
        self, optimizer, patience=100, min_lr=1e-9, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        
    
    def get_last_lr(self):
        last_lr = self.lr_scheduler.get_last_lr()
        return last_lr
        
        
class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
                
class SaveBestModel:

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    # def __call__(self, current_valid_loss, epoch, model, kernel, ode_func = None):
    def __call__(self, path, current_valid_loss, epoch, model, kernel=None, f_func=None, ode_func = None):
        if current_valid_loss < self.best_valid_loss:
            
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {epoch+1}\n")
            
            if kernel is not None: kernel_state = {'state_dict': kernel.state_dict()}
            if f_func is not None: f_func_state = {'state_dict': f_func.state_dict()}
            if ode_func is not None: ode_func_state = {'state_dict': ode_func.state_dict()}
            
            torch.save(model, os.path.join(path,'model.pt'))
            if kernel is not None: torch.save(kernel_state, os.path.join(path,'kernel.pt'))
            if f_func is not None: torch.save(f_func_state, os.path.join(path,'f_func.pt'))
            if ode_func is not None: torch.save(ode_func_state, os.path.join(path,'ode_func.pt'))
            
            
            
def load_checkpoint(path, model, optimizer, scheduler, kernel, f_func=None, ode_func = None):
    print('Loading ', os.path.join(path))
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(os.path.join(path, 'model.pt'), map_location=map_location)
    start_epoch = checkpoint['epoch']
    offset = start_epoch
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    model.load_state_dict(checkpoint['state_dict'])
    
    if kernel is not None: 
        checkpoint = torch.load(os.path.join(path, 'kernel.pt'), map_location=map_location)
        kernel.load_state_dict(checkpoint['state_dict'])
    if f_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'f_func.pt'), map_location=map_location)
        f_func.load_state_dict(checkpoint['state_dict'])
    if ode_func is not None: 
        checkpoint = torch.load(os.path.join(path, 'ode_func.pt'), map_location=map_location)
        ode_func.load_state_dict(checkpoint['state_dict'])
    
    return model, optimizer, scheduler, kernel, f_func, ode_func
                
class SaveBest_ode_Model:
    ''' have to be redone. It shoudl be using torch.save'''
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

        
    def __call__(
        self, current_valid_loss, 
        epoch, model, ode_func
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            
            pickle.dump(ode_func, open("pickled_ode_func_wODE_cos_kernel_sinF_2.pkl", "wb"))
            
class Select_times_function():
    def __init__(self,times,max_index):
        self.max_index = max_index
        self.times = times

    def select_times(self,t):
            values = torch.Tensor([])
            indices = []
            for i in range(1,t.size(0)):
                if t[i]<= self.times[self.max_index-1]:
                    values = torch.cat([values,torch.Tensor([t[i]])])
                    indices += [i]
                else:
                    pass
            return values, indices

def to_np(x):
    return x.detach().cpu().numpy()


def normalization(Data):
    for i in range(Data.size(2)):
        di = Data[:,:,i]/torch.abs(Data[:,:,i]).max()
        di = di.unsqueeze(2)
        if i == 0:
            Data_norm = di
        else:
            Data_norm = torch.cat([Data_norm,di],2)
    return Data_norm


class Integral_part():
    def __init__(self,
                 times,
                 F,
                 kernel,
                 y,
                 lower_bound = lambda x: torch.Tensor([0]).to(device),
                 upper_bound = lambda x: x,
                 MC_samplings = 10000,
                 NNs = True):
        
        self.times = times
        self.F = F
        self.kernel = kernel
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.MC_samplings = MC_samplings
        self.NNs = NNs
        self.y = y

        def _interpolate_y(y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:
            x=self.times.to(device)
            y = y.to(device)
            coeffs = natural_cubic_spline_coeffs(x, y)
            interpolation = NaturalCubicSpline(coeffs)

            def output(point:torch.Tensor):
                return interpolation.evaluate(point.to(device))

            return output
            ######################
        self.interpolated_y = _interpolate_y(y)
    
    def integral(self,x):
        
        def integrand(s):

            if self.NNs is True:
                x_aux = x.repeat(s.size(0)).view(s.size(0),1,1)
                out = self.kernel.forward(self.F.forward(self.interpolated_y(s),x),x_aux,s.unsqueeze(1))
            else:
                F_part = self.F(self.interpolated_y(s))
                out = torch.bmm(self.kernel(x,s),F_part.view(F_part.size(0),F_part.size(2),1))
              
            return out
        
        ####
        if self.lower_bound(x) < self.upper_bound(x):
            interval = [[self.lower_bound(x),self.upper_bound(x)]]
        else: 
            interval = [[self.upper_bound(x),self.lower_bound(x)]]
        ####

        return mc.integrate(
                      fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x)).to(device)*integrand(s.to(device))[:,:,:],
                       dim= 1,
                       N=self.MC_samplings,
                       integration_domain = interval, 
                       out_dim = 0
                       )
    
    def return_whole_sequence(self):
        if self.NNs is True:
            out = torch.cat([self.integral(self.times[i]) for i in range(self.times.size(0))])
        else:
            out = torch.cat([self.integral(self.times[i]).view(1,self.y.size(1)) for i in range(self.times.size(0))])
        return out
    
    
class Dynamics_Dataset_LSTM(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times, output):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        # self.frames_to_drop = frames_to_drop
        self.output = output
        # self.batch_size = batch_size

    def __getitem__(self, index):
        # print('index: ',index)
        # print('self.list_IDs.shape: ',len(self.list_IDs))
        # print('self.Data: ',self.Data)
        # print('self.times: ', self.times)
        ID = index #self.list_IDs[index]
        obs = self.Data[ID]
        output = self.output[ID]
        t = self.times #Because it already set the number of points in the main script
        # frames_to_drop = self.frames_to_drop[index]


        return obs, t, ID, output 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)
    
    
class Dynamics_Dataset3(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, Data, times, frames_to_drop, segment_len):
        'Initialization'
        self.times = times.float()
        self.Data = Data.float()
        self.frames_to_drop = frames_to_drop
        self.segment_len=segment_len
        # self.batch_size = batch_size

    def __getitem__(self, index):
        
        ID = index 
        obs = self.Data[ID,:self.segment_len]
        t = self.times 
        frames_to_drop = self.frames_to_drop[index]

        return obs, t, ID, frames_to_drop 
    
    def __len__(self):
        'Denotes the total number of points'
        return len(self.times)  
