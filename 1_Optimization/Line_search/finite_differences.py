import numpy as np

#
# Auxiliar
#
def _generate_basis_vector(X,i):
    '''
    X (numpy array): vector. Dimension (n x 1)
    i (int): subindex of entry x_i. Note: i starts at 0
    '''
    e_i = np.zeros_like(X,dtype=float)
    e_i[i]+=1
    return e_i

#
# Derivatives (centered)
#
def centered_first_derivative(f,i,X,h=1e-7):
    '''
    Computes the approximation of the function's first derivative at given X.
    Where f: f \in \R^n -> \R
    
    INPUT:
        f (function): Function to compute its first derivative.
        i (int): subindex of the variable to take the derivative, ie, x_i.
        X (numpy array): Where the first derivative computes at. Dimension (n x 1)
        h (float): Small perturbation.
    OUTPUT:
        (float): derivative of f respect to x_i at X. Dimension: \R
    '''
    # Auxiliar unit vector
    e_i = _generate_basis_vector(X,i)
    return ( f(X + e_i*h) - f(X - e_i*h) )/( 2*h )

def centered_second_derivative(f,i,j,X,h=1e-7):
    '''
    Computes the approximation of the function's second derivative at given X.
    Where f: f \in \R^n -> \R
    
    INPUT:
        f (function): Function to compute its second derivative.
        i (int): subindex of the first variable to take the derivative, ie, x_i.
        j (int): subindex of the second variable to take the derivative, ie, x_j.
        X (numpy array): Where the second derivative computes at. Dimension (n x 1)
        h (float): Small perturbation.
    OUTPUT:
        (float): second derivative of f respect to x_i at X. . Dimension: \R
    '''
    # Auxiliar unit vector
    e_i = _generate_basis_vector(X,i)
    e_j = _generate_basis_vector(X,j)
    return ( f(X + e_i*h + e_j*h) + f(X - e_i*h - e_j*h) - f(X + e_i*h - e_j*h) - f(X - e_i*h + e_j*h) )/( 4*h**2 )

#
# Gradient
#
def gradient(f,X):
    '''
    OUTPUT:
        (numpy array): approximation of the gradient of f by centered newtons method. Dimension (n x 1)
    '''
    grad = [centered_first_derivative(f,i,X) for i in range(X.shape[0])]
    return np.array(grad).reshape(X.shape)