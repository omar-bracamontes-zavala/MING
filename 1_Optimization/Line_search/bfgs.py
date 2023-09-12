import numpy as np
import finite_differences as fi
import matplotlib.pyplot as plt
import test_functions as tf
'''
    BFGS
'''

#
# Auxiliar
#
def _is_armijo_condition_satisfied(f,X_k,a_k,P_k,c1):
    return f(X_k + a_k*P_k) <= f(X_k) + c1*a_k*np.dot(fi.gradient(f,X_k).T, P_k)

def _is_curvature_condition_satisfied(f,X_k,a_k,P_k,c2):
    return -np.dot(fi.gradient(f,X_k + a_k*P_k).T, P_k) <= -c2*np.dot(fi.gradient(f,X_k).T, P_k)

def _is_strong_curvature_condition_satisfied(f,X_k,a_k,P_k,c2):
    return abs(np.dot(fi.gradient(f,X_k + a_k*P_k).T, P_k)) <= c2*abs(np.dot(fi.gradient(f,X_k).T, P_k))

def _is_wolfe_conditions_satisfied(f,X_k,a_k,P_k,take_strong=False,c1=10e-4,c2=0.9):
    armijo = _is_armijo_condition_satisfied(f,X_k,a_k,P_k,c1)
    curvature = _is_strong_curvature_condition_satisfied(f,X_k,a_k,P_k,c2) if take_strong else _is_curvature_condition_satisfied(f,X_k,a_k,P_k,c2)
    return armijo and curvature

def _compute_H0(X_old,X_new,f):
    I = np.identity(X_old.shape[0], dtype=float)
    y1 = fi.gradient(f,X_new) - fi.gradient(f,X_old)
    s1 = X_new - X_old
    
    return np.matmul(y1.T,s1) / np.matmul(y1.T,y1) * I

def _compute_H(H_k,X_old,X_new,f):
    I = np.identity(X_old.shape[0], dtype=float)
    y_k = fi.gradient(f,X_new) - fi.gradient(f,X_old)
    s_k = X_new - X_old
    rho_k = 1/np.matmul(y_k.T,s_k)
    
    W1 = I - rho_k * np.matmul(s_k, y_k.T)
    W2 = I - rho_k * np.matmul(y_k, s_k.T)    
    
    return np.matmul(np.matmul(W1,H_k), W2) + rho_k * np.matmul(s_k,s_k.T)

#
# BFGS
#
def backtracking(f,X_k,P_k,tau=1.,rho=0.9,take_strong=False,c1=1e-3,c2=0.9):
    has_optimal_step_size_found = _is_wolfe_conditions_satisfied(f,X_k,tau,P_k,take_strong,c1,c2)
    while not has_optimal_step_size_found:
        tau = rho*tau
        print('\t\t\tNew tau:',tau)
        if np.array_equal(X_k,X_k+tau*P_k):
            raise RuntimeError(f'No step size found using tau={tau}, rho={rho}')
        has_optimal_step_size_found = _is_wolfe_conditions_satisfied(f,X_k,tau,P_k,take_strong,c1,c2)
    return tau

def update_H(X_old,X_new,f,epoch,H_k=None):
    if epoch>0:
        return _compute_H(H_k,X_old,X_new,f)
    return _compute_H0(X_old,X_new,f)

def bfgs(X,f,K=10,tau=1.,rho=0.9,take_strong=False):
    '''
        BFgs algorithm using the numerical gradient
        and intelligent lenght step \alpha_k by backtracking
        
        Input:
            X (numpy array): vector with dimension nx1
            grad_f (function): Function that computes the gradient of the function. It must take just X as an mandatory input.
            K (int): Iterations to get the optimum X
            tau (float): backtracking parameter \in (0,1)
            rho (float): backtracking parameter \in (0,1)
            take_strong (bool): backtracking parameter to consider or not the strong Wolfe conditions. Default: False
            
        Output:
            X_log (list): history of the optimum X's
        
    '''
    H = np.identity(X.shape[0], dtype=float)

    X_log = [X]
    for epoch in range(K):
        # UI
        if epoch%10==0:
            print(f'\tEpoch {epoch}/{K}:')
        # Step direction (normalized)
        print('\t\tStep direction...')
        P_k = -H.dot(fi.gradient(f,X)) # We dont inv(H) since by definition H is inv(B_k)
        P_k = P_k/np.linalg.norm(P_k)
        # Step size
        print('\t\tStep size...')
        a_k = backtracking(f,X,P_k,tau,rho,take_strong)
        # New optimum X
        print('\t\tNew X...')
        X_new = X + a_k*P_k
        # Log
        X_log.append(X_new)
        # Update
        H = update_H(X,X_new,f,epoch,H)
        X = X_new
    return X_log

#
# Optimize
#
def bfgs_optimization(X0,f,K,tau=1.,rho=0.9,take_strong=False):
    bfgs_log = bfgs(X0,f,K,tau,rho,take_strong)
    bfgs_f_evaluated = [f(X) for X in bfgs_log]
    
    return bfgs_log, bfgs_f_evaluated

def plot_optimization(X0,f,K,f_name='',title='',tau=1.,rho=0.9,take_strong=False):
    print(f'\n{f_name.capitalize()}:')
    bfgs_log, bfgs_f_evaluated = bfgs_optimization(X0,f,K,tau,rho,take_strong)
    print(f'\n{f_name} BFGS: ', bfgs_log[-1],'\nf(x*)=',bfgs_f_evaluated[-1],'\n')
    
    # Plot
    plt.plot(bfgs_f_evaluated,'r-o',linewidth=1, markersize=6, alpha=0.5,label='SD')
    plt.title(title)
    plt.xlabel('Iteration k')
    plt.ylabel('$f(x_k)$')
    plt.legend()
    plt.show()

# Testing
if '__main__'==__name__:
    
    for function_name,function_params in tf.argmin_params.items():
        plot_optimization(
            function_params['x'], function_name,K=function_params['epochs'],
            title=function_params['title'],
            tau=1.,rho=0.9,take_strong=False
        )

