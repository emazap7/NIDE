#### Define solver similar to Karpel's IDE_Solver: https://github.com/JoshKarpel/idesolver/tree/master/idesolver
#### but using torch instead of numpy, using different ode solver and integration, and allowing neural networks
#### for kernel and function f (also d is allowed to take inputs x and y instead of just x).
#### The solver is also adapted to run in pure ode mode, so that it can conveniently be used to solve odes.

#General
from typing import Callable, Optional, Union
import numpy as np
import warnings
import logging
logger = logging.getLogger("idesolver")
logger.setLevel(logging.WARNING)#(logging.DEBUG)

#Torch libraries 
import torch

from torchdiffeq import odeint
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from source.integrators import MonteCarlo
mc = MonteCarlo()

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"
    


def global_error(y1: torch.Tensor, y2: torch.Tensor) -> float:
    """
    The default global error function.

    The estimate is the square root of the sum of squared differences between `y1` and `y2`.

    Parameters
    ----------
    y1 : :class:`numpy.ndarray`
        A guess of the solution.
    y2 : :class:`numpy.ndarray`
        Another guess of the solution.

    Returns
    -------
    error : :class:`float`
        The global error estimate between `y1` and `y2`.
    """
    diff = y1 - y2
    return torch.sqrt(torch.dot(diff.flatten(), diff.flatten()))



class IDESolver:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        k: Optional[Callable] = None,
        f: Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        ode_option: bool = False,
        adjoint_option=True,
        integration_dim: int = -2,
        kernel_nn: bool = False,
        ode_atol: float = 1e-5,
        ode_rtol: float = 1e-5,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        #interpolation_kind: str = "cubic",
        number_MC_samplings: int = 10000,
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
    ):
        
        
        self.y_0 = y_0
        
        self.x = x
        
        self.integration_dim = integration_dim
        
        self.n_batch = y_0.shape[0]
        
        self.number_MC_samplings = number_MC_samplings

        if c is None:
            c = lambda x, y: self._zeros()
        if d is None:
            d = lambda x, y: torch.Tensor([1])
        if k is None:
            k = lambda x, s: torch.Tensor([1])
        if f is None:
            f = lambda y: self._zeros()
         
        if ode_option is True:
            k = lambda x,s: torch.Tensor([0])
        if ode_option is True:
            f = lambda y: self._zeros()
            
        self.c = lambda x, y: c(x, y)
        self.d = lambda x, y: d(x,y)
        if kernel_nn is True:
            self.k = k
        else:
            self.k = lambda x, s: k(x, s)
        self.f = lambda y: f(y)
        
        self.kernel_nn = kernel_nn
        self.ode_option = ode_option
        self.adjoint_option = adjoint_option
        
                
        
        if lower_bound is None:
            lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be 0 if max_iterations is None")
        if global_error_tolerance < 0:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
        self.global_error_tolerance = global_error_tolerance
        self.global_error_function = global_error_function
        
        
        #self.interpolation_kind = interpolation_kind

        if not 0 < smoothing_factor < 1:
            raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")
        self.smoothing_factor = smoothing_factor

        
        
        if max_iterations is not None and max_iterations <= 0:
            raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")
        
        
        self.max_iterations = max_iterations

        #self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        self.int_atol = int_atol
        self.int_rtol = int_rtol

        self.store_intermediate = store_intermediate_y
        if self.store_intermediate:
            self.y_intermediate = []

        self.iteration = None
        self.y = None
        self.global_error = None
        
    def _zeros(self) -> torch.Tensor:
        return torch.zeros_like(self.y_0)    
    
    
    
    def solve(self, callback: Optional[Callable] = None) -> torch.Tensor:
            
        with warnings.catch_warnings():
                    warnings.filterwarnings(
                    action="error",
                    message="Casting complex values",
                    category=np.ComplexWarning,
                    )
            
            
            #try:
                    y_current = self._initial_y()
                    if self.ode_option is True:
                        y_guess = y_current
                    else:
                        y_guess = self._solve_rhs_with_known_y(y_current) 
                    error_current = self._global_error(y_current, y_guess)
                    
                    #if self.store_intermediate:
                    #    self.y_intermediate.append(y_current)
                    
                    if self.ode_option is not True:
                        self.iteration = 0


                        logger.debug(
                            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                        )

                        if callback is not None:
                            logger.debug(f"Calling {callback} after iteration {self.iteration}")
                            callback(self, y_guess, error_current)

                        while error_current > self.global_error_tolerance:


                            new_current = self._next_y(y_current, y_guess)
                            new_guess = self._solve_rhs_with_known_y(new_current)
                            new_error = self._global_error(new_current, new_guess)
                            if new_error > error_current:
                                warnings.warn(
                                    f"Error increased on iteration {self.iteration}",
                                    #exceptions.IDEConvergenceWarning,
                                )

                            y_current, y_guess, error_current = (
                                new_current,
                                new_guess,
                                new_error,
                                )

                            if self.store_intermediate:
                                self.y_intermediate.append(y_current)

                            self.iteration += 1


                            logger.debug(
                            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                            )

                            if callback is not None:
                                logger.debug(f"Calling {callback} after iteration {self.iteration}")
                                callback(self, y_guess, error_current)

                            if self.max_iterations is not None and self.iteration >= self.max_iterations:
                            
                                break


        
        
        self.y = y_guess
        self.global_error = error_current


        return self.y
    
    
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        
        return self._solve_ode(self.c)

    
    
    
    def _next_y(self, curr: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    
    
    
    def _global_error(self, y1: torch.Tensor, y2: torch.Tensor) -> float:
        """
        Return the global error estimate between `y1` and `y2`.

        Parameters
        ----------
        y1
            A guess of the solution.
        y2
            Another guess of the solution.

        Returns
        -------
        error : :class:`float`
            The global error estimate between `y1` and `y2`.
        """
        return self.global_error_function(y1, y2)

    
    
    
    def _solve_rhs_with_known_y(self, y: torch.Tensor) -> torch.Tensor:
        """Solves the right-hand-side of the IDE as if :math:`y` was `y`."""
        interpolated_y = self._interpolate_y(y)
        
        #mc = MonteCarlo()

        def integral(x):
            x = x.to(device)
            
            def integrand(s):
                
                if self.adjoint_option is True:
                    s = s.to(device)
                else:
                    s=s.to(device).requires_grad_(True)
                if self.kernel_nn is False:
                    out = torch.bmm(self.k(x, s), self.f(interpolated_y(s[:]))\
                                    .reshape(self.n_batch,self.number_MC_samplings,self.y_0.shape[-1],1))
                else:
                    y_in = self.f(interpolated_y(s[:]))\
                            .reshape(self.n_batch,self.number_MC_samplings,self.y_0.shape[-1])
                    out = self.k.forward(y_in,x.repeat(self.number_MC_samplings).reshape(self.number_MC_samplings,1),s)
                    
                return out
                
            ####
            if self.lower_bound(x) < self.upper_bound(x):
                interval = [[self.lower_bound(x),self.upper_bound(x)]]
            else: 
                interval = [[self.upper_bound(x),self.lower_bound(x)]]
            ####
            
            
            return mc.integrate(
                           fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x))*integrand(s),
                           dim= 1,
                           N=self.number_MC_samplings,
                           integration_domain = interval, 
                           out_dim = self.integration_dim,
                           )
        
        
        

        def rhs(x, y):
            if self.adjoint_option is True:
                x = x.to(device)
            else:
                x=x.to(device).requires_grad_(True)
            if self.adjoint_option is True:
                y = y.to(device)
            else:
                y=y.to(device).requires_grad_(True)
            
            return self.c(x, interpolated_y(x).to(device)).to(device) + (self.d(x,interpolated_y(x).to(device)).to(device)*integral(x).to(device)).to(device)
            
        return self._solve_ode(rhs)

    
    
    def _interpolate_y(self, y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            The y values to interpolate (probably a guess at the solution).

        Returns
        -------
        interpolator : :class:`scipy.interpolate.interp1d`
            The interpolator function.
        """
        x=self.x
        y = y
        
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point.to(device))
        
        return output
        

    
    
    def _solve_ode(self, rhs: Callable) -> torch.Tensor:
        """Solves an ODE with the given right-hand side."""
        
        
        fun = rhs
        if self.adjoint_option is True or self.ode_option is False:
            y0 = self.y_0
        else:
            y0=self.y_0.requires_grad_(True)
        if self.adjoint_option is False or self.ode_option is False:
            t_span= torch.linspace(self.x[0],self.x[-1],self.x.size(0)).to(device)
        else:
            t_span= torch.linspace(self.x[0],self.x[-1],self.x.size(0)).to(device).requires_grad_(True)
        
        
        sol = odeint(fun,y0,t_span,rtol=self.ode_rtol, atol=self.ode_atol, method = 'dopri5')#,options=dict(step_size=1e-5))
        sol = sol.permute(1,0,2)
        
        return sol
    
    
    
class IDESolver_monoidal:
    
    def __init__(
        self,
        x: torch.Tensor,
        y_0: Union[float, np.float64, complex, np.complex128, np.ndarray, list,torch.Tensor,torch.tensor],
        c: Optional[Callable] = None,
        d: Optional[Callable] = None,
        k: Optional[Callable] = None,
        f: Optional[Callable] = None,
        lower_bound: Optional[Callable] = None,
        upper_bound: Optional[Callable] = None,
        global_error_tolerance: float = 1e-6,
        max_iterations: Optional[int] = None,
        ode_option: bool = False,
        adjoint_option=True,
        integration_dim: int = 0,
        kernel_nn: bool = False,
        ode_atol: float = 1e-5,
        ode_rtol: float = 1e-5,
        int_atol: float = 1e-5,
        int_rtol: float = 1e-5,
        #interpolation_kind: str = "cubic",
        smoothing_factor: float = 0.5,
        store_intermediate_y: bool = False,
        global_error_function: Callable = global_error,
    ):
        
        
        self.y_0 = y_0
        
        self.x = x
        
        self.integration_dim = integration_dim

        if c is None:
            c = lambda x, y: self._zeros()
        if d is None:
            d = lambda x, y: torch.Tensor([1])
        if k is None:
            k = lambda x, s: torch.Tensor([1])
        if f is None:
            f = lambda y: self._zeros()
         
        if ode_option is True:
            k = lambda x,s: torch.Tensor([0])
        if ode_option is True:
            f = lambda y: self._zeros()
            
        self.c = lambda x, y: c(x, y)
        self.d = lambda x, y: d(x,y)
        if kernel_nn is True:
            self.k = k
        else:
            self.k = lambda x, s: k(x, s)
        self.f = lambda y: f(y)
        
        self.kernel_nn = kernel_nn
        self.ode_option = ode_option
        self.adjoint_option = adjoint_option
        
                
        
        if lower_bound is None:
            lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        
        
        if global_error_tolerance == 0 and max_iterations is None:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be 0 if max_iterations is None")
        if global_error_tolerance < 0:
            raise exceptions.InvalidParameter("global_error_tolerance cannot be negative")
        self.global_error_tolerance = global_error_tolerance
        self.global_error_function = global_error_function
        
        
        #self.interpolation_kind = interpolation_kind

        if not 0 < smoothing_factor < 1:
            raise exceptions.InvalidParameter("Smoothing factor must be between 0 and 1")
        self.smoothing_factor = smoothing_factor

        
        
        if max_iterations is not None and max_iterations <= 0:
            raise exceptions.InvalidParameter("If given, max iterations must be greater than 0")
        
        
        self.max_iterations = max_iterations

        #self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol

        self.int_atol = int_atol
        self.int_rtol = int_rtol

        self.store_intermediate = store_intermediate_y
        if self.store_intermediate:
            self.y_intermediate = []

        self.iteration = None
        self.y = None
        self.global_error = None
        
    def _zeros(self) -> torch.Tensor:
        return torch.zeros_like(self.y_0)    
    
    
    
    def solve(self, callback: Optional[Callable] = None) -> torch.Tensor:
            
        with warnings.catch_warnings():
                    warnings.filterwarnings(
                    action="error",
                    message="Casting complex values",
                    category=np.ComplexWarning,
                    )
            
            
            #try:
                    y_current = self._initial_y()
                    if self.ode_option is True:
                        y_guess = y_current
                    else:
                        y_guess = self._solve_rhs_with_known_y(y_current) 
                    error_current = self._global_error(y_current, y_guess)
                    
                    #if self.store_intermediate:
                    #    self.y_intermediate.append(y_current)
                    
                    if self.ode_option is not True:
                        self.iteration = 0


                        logger.debug(
                            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                        )

                        if callback is not None:
                            logger.debug(f"Calling {callback} after iteration {self.iteration}")
                            callback(self, y_guess, error_current)

                        while error_current > self.global_error_tolerance:


                            new_current = self._next_y(y_current, y_guess)
                            new_guess = self._solve_rhs_with_known_y(new_current)
                            new_error = self._global_error(new_current, new_guess)
                            if new_error > error_current:
                                warnings.warn(
                                    f"Error increased on iteration {self.iteration}",
                                    #exceptions.IDEConvergenceWarning,
                                )

                            y_current, y_guess, error_current = (
                                new_current,
                                new_guess,
                                new_error,
                                )

                            if self.store_intermediate:
                                self.y_intermediate.append(y_current)

                            self.iteration += 1


                            logger.debug(
                            f"Advanced to iteration {self.iteration}. Current error: {error_current}."
                            )

                            if callback is not None:
                                logger.debug(f"Calling {callback} after iteration {self.iteration}")
                                callback(self, y_guess, error_current)

                            if self.max_iterations is not None and self.iteration >= self.max_iterations:
                            
                                break


        
        
        self.y = y_guess
        self.global_error = error_current


        return self.y
    
    
    
    
    def _initial_y(self) -> torch.Tensor:
        """Calculate the initial guess for `y`, by considering only `c` on the right-hand side of the IDE."""
        return self._solve_ode(self.c)

    
    
    
    def _next_y(self, curr: torch.Tensor, guess: torch.Tensor) -> torch.Tensor:
        """Calculate the next guess at the solution by merging two guesses."""
        return (self.smoothing_factor * curr) + ((1 - self.smoothing_factor) * guess)

    
    
    
    def _global_error(self, y1: torch.Tensor, y2: torch.Tensor) -> float:
        """
        Return the global error estimate between `y1` and `y2`.

        Parameters
        ----------
        y1
            A guess of the solution.
        y2
            Another guess of the solution.

        Returns
        -------
        error : :class:`float`
            The global error estimate between `y1` and `y2`.
        """
        return self.global_error_function(y1, y2)

    
    
    
    def _solve_rhs_with_known_y(self, y: torch.Tensor) -> torch.Tensor:
        """Solves the right-hand-side of the IDE as if :math:`y` was `y`."""
        interpolated_y = self._interpolate_y(y)
        
        #mc = MonteCarlo()

        def integral(x):
            number_MC_samplings = 1000
            x = x.to(device)
            
            def integrand(s):
                
                if self.adjoint_option is True:
                    s = s.to(device)
                else:
                    s=s.to(device).detach().requires_grad_(True)
                if self.kernel_nn is False:
                    out = torch.bmm(self.k(x, s), self.f(interpolated_y(s[:])).reshape(number_MC_samplings,self.y_0.size(0),1))
                else:
                    y_in = self.f(interpolated_y(s[:])).reshape(number_MC_samplings,self.y_0.size(0))
                    out = self.k.forward(y_in,x.repeat(number_MC_samplings).reshape(number_MC_samplings,1),s)
                    out = out.unsqueeze(2)
                return out
                
            ####
            if self.lower_bound(x) < self.upper_bound(x):
                interval = [[self.lower_bound(x),self.upper_bound(x)]]
            else: 
                interval = [[self.upper_bound(x),self.lower_bound(x)]]
            ####
            

            return mc.integrate(
                           fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x))*integrand(s)[:,:self.y_0.size(0),0],
                           dim= 1,
                           N=number_MC_samplings,
                           integration_domain = interval, 
                           out_dim = self.integration_dim,
                           )
        
        
        

        def rhs(x, y):
            if self.adjoint_option is True:
                x = x.to(device)
            else:
                x=x.to(device).detach().requires_grad_(True)
            if self.adjoint_option is True:
                y = y.to(device)
            else:
                y=y.to(device).detach().requires_grad_(True)
            
            return self.c(x, interpolated_y(x).to(device)).to(device) + (self.d(x,interpolated_y(x).to(device)).to(device)*integral(x).to(device)).to(device)
            
        return self._solve_ode(rhs)

    
    
    def _interpolate_y(self, y: torch.Tensor):# -> torch.Tensor: #inter.interp1d:
        """
        Interpolate `y` along `x`, using `interpolation_kind`.

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            The y values to interpolate (probably a guess at the solution).

        Returns
        -------
        interpolator : :class:`scipy.interpolate.interp1d`
            The interpolator function.
        """
        x=self.x
        y = y
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point.to(device))
        
        return output
        

    
    
    def _solve_ode(self, rhs: Callable) -> torch.Tensor:
        """Solves an ODE with the given right-hand side."""
        
        
        fun = rhs
        if self.adjoint_option is True or self.ode_option is False:
            y0 = self.y_0
        else:
            y0=self.y_0.detach().requires_grad_(True)
        if self.adjoint_option is False or self.ode_option is False:
            t_span= torch.linspace(self.x[0],self.x[-1],self.x.size(0)).to(device)
        else:
            t_span= torch.linspace(self.x[0],self.x[-1],self.x.size(0)).to(device).detach().requires_grad_(True)
        
        
        sol = odeint(fun,y0,t_span,rtol=self.ode_rtol, atol=self.ode_atol, method = 'dopri5')#,options=dict(step_size=1e-5))

        return sol
    
