import numpy as np
#
# Test Functions
#
def sphere(X,c=1.0):
    '''
        Translated Sphere's function
            f = f(x): \R^n --> \R
        
        Input:
            X (numpy array): vector with dimension nx1
            c (float): The center of the sphere. It is assumed that the center would be an uniform vector.
        Output:
            (float): The value of the function evaluated on X
        
    '''
    C = np.full_like(X,c, dtype=float)
    return np.sum(np.square(np.subtract(X,C)))
def rosenbrock(X):
    '''
        Rosenbrock's function
            f = f(x): \R^n --> \R
        
        Input:
            X (numpy array): vector with dimension nx1
        Output:
            (float): The value of the function evaluated on X
        
    '''
    return np.sum([100 * (X[i+1] - X[i]**2)**2 + (X[i] - 1)**2 for i in range(X.shape[0]-1)])
def perm(X,b=1.):
    '''
        Perm's function
            f = f(x): \R^n --> \R
        
        Input:
            X (numpy array): vector with dimension nx1
            b (float): just a perm's constant
        Output:
            (float): The value of the function evaluated on X
        
    '''
    dim = X.shape[0]
    def _j_sum_addend(x_j,i,j):
        return (j**i + b) * ( ( x_j/j )**i - 1) 
    def _j_sum(X,i):
        return sum([ _j_sum_addend(X[j-1],i,j)for j in range(1,dim+1)])
    
    return sum([ _j_sum(X,i)**2 for i in range(1,dim+1)])

def perm_2(X, b=1.):
    dim = X.shape[0]
    result = 0

    for i in range(1, dim+1):
        inner = 0
        for j in range(1, dim+1):
            inner += ((j) **(i) + b) * ((X[j-1] / j) ** (i) - 1)
        result += inner ** 2

    return result
# Initialize test parameters
dimension = 5
X0 = np.full((dimension,),0.5, dtype=float)
argmin_params = {
    'sphere':{
        'f':sphere,# Function
        'x':X0, # Numpy array
        'epochs':10,
        'step_size':0.5,
        'title':'Translated Sphere in $R^n$',
    },
    'rosen':{
        'f':rosenbrock,# Function
        'x':X0, # Numpy array
        'epochs':50, #100000 pero es muy lento
        'step_size':1e-3,
        'title':'Rosenbrock in $R^n$',
    },
    'perm':{
        'f':perm_2,# Function
        'x':X0, # Numpy array
        'epochs':1000, #300000 pero es muy lento
        'step_size':1e-8,
        'title':'Perm in $R^n$',
    },
}
