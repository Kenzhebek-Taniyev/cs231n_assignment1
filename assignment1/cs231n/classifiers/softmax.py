from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dot = np.zeros((X.shape[0], W.shape[1]))
    
    for i in range(X.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[0]):
                dot[i][j] += X[i][k] * W[k][j]
        dot[i, :] = np.exp(dot[i, :])
        dot[i, :] /= np.sum(dot[i, :])  
        loss -= np.log(dot[i, y[i]]) 
        dot[i, y[i]] -= 1
    
    for i in range(X.shape[0]):
        for j in range(dot.shape[0]):
            for k in range(dot.shape[1]):
                dW[j][k] += X[i][j] * W[i][k]
            
    loss /= X.shape[0]
    loss += 0.5*reg*np.sum(W**2)
    
    dW /= X.shape[0]
    dW += reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dot = X.dot(W)
    dot = np.exp(dot)
    dot /= np.sum(dot, axis=1, keepdims=True)
    loss -= np.sum(np.log(dot[range(dot.shape[0]), y]))
    
    dot[range(dot.shape[0]), y] -= 1
    dW = X.T.dot(dot)
    
    loss /= X.shape[0]
    loss += 0.5*reg*np.sum(W**2)
    
    dW /= X.shape[0]
    dW += reg*W
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
