#General libraries
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import scipy
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

#Custom libraries
from utils_ide import Select_times_function, EarlyStopping, SaveBestModel, to_np, Integral_part, LRScheduler, load_checkpoint, Train_val_split, Dynamics_Dataset, Test_Dynamics_Dataset
from torch.utils.data import SubsetRandomSampler
from source.solver import IDESolver, IDESolver_monoidal

#Torch libraries
import torch
from torch.nn import functional as F


def IDE_spiral_experiment_no_adjoint(kernel, F_func, ode_func, Data, dataloaders, time_seq, ode_nn, extrapolation_points, args): 
        
    # -- metadata for saving checkpoints
    if args.model=='nide': 
        str_model_name = "nide"
    elif args.model=='node': 
        str_model_name = "node"
    
    str_model = f"{str_model_name}"
    str_log_dir = args.root_path
    path_to_experiment = os.path.join(str_log_dir,str_model_name, args.experiment_name)

    if args.mode=='train':
        if not os.path.exists(path_to_experiment):
            os.makedirs(path_to_experiment)

        
        print('path_to_experiment: ',path_to_experiment)
        txt = os.listdir(path_to_experiment)
        if len(txt) == 0:
            num_experiments=0
        else: 
            num_experiments = [int(i[3:]) for i in txt]
            num_experiments = np.array(num_experiments).max()
         # -- logger location
        writer = SummaryWriter(os.path.join(path_to_experiment,'run'+str(num_experiments+1)))
        print('writer.log_dir: ',writer.log_dir)
        
        path_to_save_plots = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'plots')
        path_to_save_models = os.path.join(path_to_experiment,'run'+str(num_experiments+1),'model')
        if not os.path.exists(path_to_save_plots):
            os.makedirs(path_to_save_plots)
        if not os.path.exists(path_to_save_models):
            os.makedirs(path_to_save_models)
            
        with open(os.path.join(writer.log_dir,'commandline_args.txt'), 'w') as f:
            for key, value in args.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))



    obs = Data
    times = time_seq
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    loader_test = dataloaders['test']
    # Train Neural IDE
    get_times = Select_times_function(times,extrapolation_points)

    
    if ode_nn is True:
        All_parameters = list(kernel.parameters()) + list(F_func.parameters()) + list(ode_func.parameters())
    else:
         All_parameters = list(kernel.parameters()) + list(F_func.parameters())
            
    
    optimizer = torch.optim.Adam(All_parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.plat_patience, min_lr=args.min_lr, factor=args.factor)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr,last_epoch=-1)
    
    
    if args.resume_from_checkpoint is not None:
        path = os.path.join(args.root_path,args.model,args.experiment_name,args.resume_from_checkpoint,'model')
        print(path)
        _, optimizer, scheduler, kernel, F_func, ode_func = load_checkpoint(path, kernel, optimizer, scheduler, kernel, F_func, ode_func)


    
    if args.mode=='train':
        
        early_stopping = EarlyStopping(patience=1000,min_delta=0)

        
        all_train_loss=[]
        all_val_loss=[]

        save_best_model = SaveBestModel()
        start = time.time()
        for i in range(args.epochs):
            kernel.train()
            F_func.train()
            if ode_nn is True:
                ode_func.train()
            start_i = time.time()
            print('Epoch:',i)
            
            counter=0
            train_loss = 0.0
            
            for obs_, ts_, ids_ in tqdm(train_loader): 
                obs_ = obs_.to(args.device).detach()
                ts_ = ts_.to(args.device)
                ids_ = ids_.to(args.device)
                


                y_0 = obs_[:,0,:]
                
                z_ = IDESolver(time_seq,
                                        y_0,
                                        c = ode_func,
                                        k = kernel,
                                        f = F_func,
                                        lower_bound = args.alpha,
                                        upper_bound = args.beta,
                                        max_iterations = 3,
                                        ode_option = False,
                                        adjoint_option = False,
                                        integration_dim = -2,
                                        kernel_nn = True,
                                        number_MC_samplings=args.number_MC_samplings
                                        ).solve()


                loss = F.mse_loss(z_, obs_.detach())  



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                counter += 1
                train_loss += loss.item()
                    
                del obs_, z_, y_0, ts_, ids_
                

            train_loss /= counter
            all_train_loss.append(train_loss)
            
            if args.lr_scheduler == 'CosineAnnealingLR' and i>args.warmup:
                scheduler.step()
            elif args.lr_scheduler != 'CosineAnnealingLR':
                scheduler.step(train_loss)
            

            ## Validating
            kernel.eval()
            F_func.eval()
            if ode_nn is True:
                ode_func.eval()
            with torch.no_grad():

                
                val_loss = 0.0
                counter = 0
                if len(val_loader)>0:
                    
                    for obs_val, ts_val, ids_val in tqdm(val_loader):
                        obs_val = obs_val.to(args.device)
                        ts_val = ts_val.to(args.device)
                        ids_val = ids_val.to(args.device)

                        ids_val, indices = torch.sort(ids_val)
                        

                        
                        y_0 = obs_val[:,0,:]
                
                        z_val = IDESolver(time_seq,
                                                y_0,
                                                c = ode_func,
                                                k = kernel,
                                                f = F_func,
                                                lower_bound = args.alpha,
                                                upper_bound = args.beta,
                                                max_iterations = 3,
                                                ode_option = False,
                                                adjoint_option = False,
                                                integration_dim = -2,
                                                kernel_nn = True,
                                                number_MC_samplings=args.number_MC_samplings
                                                ).solve()
                        
                        loss_validation = F.mse_loss(z_val, obs_val.detach())
                        

                        counter += 1
                        val_loss += loss_validation.item()
                        
                        del obs_val, z_val, y_0, ts_val, ids_val
                else: counter += 1

                val_loss /= counter
                all_val_loss.append(val_loss)

                writer.add_scalar('train_loss', all_train_loss[-1], global_step=i)
                if len(all_val_loss)>0:
                    writer.add_scalar('val_loss', all_val_loss[-1], global_step=i)
                if args.lr_scheduler == 'ReduceLROnPlateau':
                    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], global_step=i)
                elif args.lr_scheduler == 'CosineAnnealingLR':
                    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=i)


                if i % args.plot_freq == 0:
                    obs_test, ts_test, ids_test = next(iter(loader_test))

                    ids_test, indices = torch.sort(ids_test)
                    
                    


                    obs_test = obs_test.to(args.device)
                    ts_test = ts_test.to(args.device)
                    ids_test = ids_test.to(args.device)
                    
                    y_0 = obs_test[:,0,:]
                
                    z_test = IDESolver(time_seq,
                                            y_0,
                                            c = ode_func,
                                            k = kernel,
                                            f = F_func,
                                            lower_bound = args.alpha,
                                            upper_bound = args.beta,
                                            max_iterations = 3,
                                            ode_option = False,
                                            adjoint_option = False,
                                            integration_dim = -2,
                                            kernel_nn = True,
                                            number_MC_samplings=args.number_MC_samplings
                                            ).solve()
                    
                    plt.figure(0, facecolor='w')
                    
                    plt.plot(np.log10(all_train_loss))
                    plt.plot(np.log10(all_val_loss))
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE Loss")
                    
                    plt.savefig(os.path.join(path_to_save_plots,'losses'))

                    new_times = to_np(ts_test)

                    plt.figure(figsize=(8,8),facecolor='w')
                    z_p = z_test
                    z_p = to_np(z_p)

                    plt.figure(1, facecolor='w')
                    plt.plot(z_p[0,:,0],z_p[0,:,1],c='r', label='model')
                    obs_print = to_np(obs_test)
                    
                    plt.scatter(obs_print[0,:,0],obs_print[0,:,1],label='Data',c='blue', alpha=0.5)
                    plt.xlabel("dim 0")
                    plt.ylabel("dim 1")
                    
                    plt.legend()
                    
                    plt.savefig(os.path.join(path_to_save_plots,'plot_dim0vsdim1_epoch'+str(i)))


                    plt.close('all')
                    
                    del obs_test, z_test, y_0, ts_test, ids_test

            end_i = time.time()
            

            model_state = {
                    'epoch': i + 1,
                    'state_dict': kernel.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
            }


            if len(val_loader)>0: 
                save_best_model(path_to_save_models, all_val_loss[-1], i, model_state, kernel, F_func, ode_func)
            else: 
                save_best_model(path_to_save_models, all_train_loss[-1], i, model_state, kernel, F_func, ode_func)


            early_stopping(all_val_loss[-1])
            if early_stopping.early_stop:
                break


        end = time.time()
        
    elif args.mode=='evaluate':
        print('Running in evaluation mode')
        
        kernel.eval()
        F_func.eval()
        if ode_nn is True:
            ode_func.eval()
        with torch.no_grad():
            for obs_test, ts_test, ids_test in tqdm(loader_test):

                ids_test, indices = torch.sort(ids_test)



                obs_test = obs_test.to(args.device)
                ts_test = ts_test.to(args.device)
                ids_test = ids_test.to(args.device)

                y_0 = obs_test[:,0,:]

                z_test = IDESolver(time_seq,
                                        y_0,
                                        c = ode_func,
                                        k = kernel,
                                        f = F_func,
                                        lower_bound = args.alpha,
                                        upper_bound = args.beta,
                                        max_iterations = 3,
                                        ode_option = False,
                                        adjoint_option = False,
                                        integration_dim = -2,
                                        kernel_nn = True,
                                        number_MC_samplings=args.number_MC_samplings
                                        ).solve()

            
            z_p = to_np(z_test)
            obs_print = to_np(obs_test)
            
             
            data_to_plot = obs_print  
            predicted_to_plot = z_p
            
            
            plt.figure(figsize=(10,10),dpi=200,facecolor='w')
            plt.scatter(data_to_plot[0,:,0],data_to_plot[0,:,1],label='Data')
            plt.plot(predicted_to_plot[0,:,0],predicted_to_plot[0,:,1],label='Model',c='red',linewidth=3)
            plt.xlabel("dim 0",fontsize=20)
            plt.ylabel("dim 1",fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20) 
            
            

            all_r2_scores = []
            all_mse_scores = []

            for idx_frames in range(len(data_to_plot)):
                _, _, r_value, _, _ = scipy.stats.linregress(data_to_plot[idx_frames,...].flatten(), predicted_to_plot[idx_frames,...].flatten())
                all_r2_scores.append(r_value)
                
                tmp_mse_loss = mean_squared_error(data_to_plot[idx_frames,...].flatten(), predicted_to_plot[idx_frames,...].flatten())
                all_mse_scores.append(tmp_mse_loss)

            
            print('R2: ',all_r2_scores)
            print('MSE: ',all_mse_scores)
            
            print('R2: ',all_r2_scores)
            print('MSE: ',all_mse_scores)
            
            _, _, r_value_seq, _, _ = scipy.stats.linregress(data_to_plot[:,:].flatten(), predicted_to_plot[:,:].flatten())
            mse_loss = mean_squared_error(data_to_plot[:,:].flatten(), predicted_to_plot[:,:].flatten())
            
            print('R2:',r_value_seq)
            print('MSE:',mse_loss)
        
        
