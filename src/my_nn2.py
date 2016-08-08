import numpy as np
import collections


# Write a function that constrains data to be between -1 and 1
# This is important for constraining data before running through the neural network
# Assumes data is N_training_examples x N_variables
def norm(x):
    # Take extremes for each variable in x 
    xmax = np.amax(x, axis=0)
    xmin = np.amin(x, axis=0)

    # Normalize data (careful to avoid dividing by zero)
    x = np.where(xmax - xmin != 0, (2*x - xmax - xmin) / (xmax - xmin), 0)

    # Output tuple
    return x, xmin, xmax
    
# This function unnormalizes data, by doing the reverse operation of "norm"
def unnorm(x, xmin, xmax):
    # Apply reverse norm
    x = (x * (xmax - xmin) + xmax + xmin) / 2.0

    # Output unnormalized data
    return x

def stdize(x):
    xmean = np.mean(x, axis=0)
    xstd  = np.std( x, axis=0)

    x = np.where(xstd != 0, ( x - xmean ) / xstd, 0)

    return x, xmean, xstd

def unstdize(x,xmean,xstd):

    x = x * xstd + xmean

    return x

# Forward propagation algorithm
def fwd_prop(X, w2, b2, w3, b3):
    # Forward propagation
    z2 = np.dot(X,w2) + b2
    a2 = np.tanh(z2) #activation function
    z3 = np.dot(a2,w3) + b3
    a3 = np.tanh(z3) #activation function
    return a3, a2 

# Backpropagation algorithm to calculate errors
def back_prop(X, y, a3, a2, w3):
    # Calculate deltas - depends on tanh activation and lsqs error
    delta3 = -(y - a3) * (1 -np.square(a3))
    delta2 = np.dot(delta3, w3.T) * (1 - np.square(a2))

    # Calculate dC/db (partial deriv)
    db3 = np.sum(delta3, axis=0, keepdims=False) #sum over all N_cases, now 1xm
    db2 = np.sum(delta2, axis=0, keepdims=False) #ditto, now 1xh
  
    # Calculate dC/dw (partial deriv)
    dw3 = np.dot(a2.T, delta3)
    dw2 = np.dot(X.T , delta2)

    # Return partial derivatives
    return dw3, dw2, db3, db2


# This function learns parameters for the neural network and returns the model.
# - h: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
# learning rate for gradient descent
# regularization strength
def run(X, y, h, num_passes=25000, print_loss=False, regularization=0.1, learning_rate=0.01):

    minibatch = True

    # Initialize named tuple for output data
    out = collections.namedtuple('out', ['w2', 'b2', 'w3', 'b3'])

        
   
    # Get number of input and output variables
    N = X.shape[0] # = Yout.shape[0]
    n = X.shape[1]
    m = y.shape[1]


    if minibatch:
        N = int(N / 10)
        num_passes = int(num_passes / 10)
        minibatch_loops = 10
    else:
        minibatch_loops = 1

    # Initialize the parameters to random values. We need to learn these.
    # np.random.seed(2)
    w2 = np.random.randn(n, h) / np.sqrt(n)
    b2 = np.zeros((1,h)) #np.random.randn(1, h) #maybe initialize as zeros?
    w3 = np.random.randn(h, m) / np.sqrt(h)
    b3 = np.zeros((1,m)) #np.random.randn(1, m)

  
    # Gradient descent. For each batch...
    for i in range(num_passes):

        for j in range(minibatch_loops):
            ind = np.arange(j*N, (j+1)*N)
            Xin = X[ind,:]
            yin = y[ind,:]
   

            # Forward propagation to find solution and store hidden layer output
            a3, a2 = fwd_prop(Xin, w2, b2, w3, b3)       

            # Backpropagation
            dw3, dw2, db3, db2 = back_prop(Xin, yin, a3, a2, w3)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dw3 += regularization * w3
            dw2 += regularization * w2

            # Update model parameters using gradient descent
            w2 += -learning_rate * dw2
            b2 += -learning_rate * db2
            w3 += -learning_rate * dw3
            b3 += -learning_rate * db3

        
        # Print loss/cost to show progress (expensive)
        if print_loss and (i % (num_passes/20) == 0.0 or i < 10 or i==num_passes-1):
	    # Calculate loss function
            yout, _ = fwd_prop(X, w2, b2, w3, b3)
            loss = (1/(2.*N)) * np.sum(np.square(y - yout))
            print("Loss after iteration %i: %f" %(i, loss) )

    # Calculate a3 (yout) on full batch
    yout, _ = fwd_prop(X, w2, b2, w3, b3)

    # Assign new parameters to the model
    output = out(w2 = w2, b2 = b2, w3 = w3, b3 = b3)
        
    return yout, output


