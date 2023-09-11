import numpy as np
import finite_differences as fi

#
# Hessian
#
def _hessian(f,X):
    '''
    OUTPUT:
        (numpy array): approximation of the gradient of f by centered newtons method. Dimension: \R^{n \cross n}
    '''
    hess_list = [fi.centered_second_derivative(f,i,j,X) for i,_ in enumerate(X) for j,_ in enumerate(X)]
    return np.array(hess_list).reshape(X.shape[0],X.shape[0])

#
# Newtons Method
#
def newtons_method(f,X,K=500):
    '''
    INPUT:
        f (function): target function to locally optimize.
        X (numpy array): user's starting point. Dimension: \R^n
    OUTPUT:
        (list): Log of all X_new that locally optimize f. Note that the last X_new in this log should be the local optimum (approximation).
    '''
    X_log = [X]
    for _ in range(K):
        # Prev
        grad_f = fi.gradient(f,X)
        hess_f = _hessian(f,X)
        hess_f_inv = np.linalg.inv(hess_f)
        # Newton's Method
        X_new = X - hess_f_inv.dot(grad_f)
        # Log
        X_log.append(X_new)
        # Update
        X = X_new
        
        # Termination criteria
        if does_termination_criteria_1_reached(f,X_log):
            break
    
    return X_log
   
    
# Testing
if __name__=='__main__':
    def f(X):
        return X[0]**2 + 3*X[1]**2
    
    X = np.array([1,1])
    a = fi.gradient(f,X)
    b = _hessian(f,X)
    print('Grad:', a)
    print('Hess:', b)
    
    X_log = newtons_method(f,X)
    print(X_log[-1]) # Should be X_optimum=(0,0)
    optimum_f = f(X_log[-1])
    print(optimum_f) # Should be f(X_optimum)=0
    
