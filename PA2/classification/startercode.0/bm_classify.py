import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        #change all the zeros to 1
        y[y == 0] = -1
        #weight bias matrix
        wb = np.append(w, b)
        #adding for bias
        X = np.column_stack((X, np.array([1]*N)))
        for i in range(max_iterations):
            wTx = np.dot(X, wb)
            #y.wT. x
            result = y * wTx
            #pick those with less <= 0.. np.int64 is used to get 1 or 0
            filterResults = np.int64(result <= 0)
            #this is the update value..
            update = np.dot(X.transpose(), filterResults * y)
            wb = wb + ((step_size/N) * update)




    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        ############################################
        # change all the zeros to 1
        y[y == 0] = -1
        # weight bias matrix
        wb = np.append(w, b)
        # adding for bias
        X = np.column_stack((X, np.array([1] * N)))
        for i in range(max_iterations):
            wTx = np.dot(X, wb)
            exp = -1 * y * wTx
            #apply sigmoid for logistic regression
            result = sigmoid(exp)
            update = np.dot(X.transpose(), result * y)
            wb = wb + ((step_size/N) * update)
    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    w = wb[:D]
    b = wb[D]
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = z
    ############################################
    value = 1/(1+np.exp(-value))
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        predictions =  np.dot(X, w) + b
        #prediction is >0 or <= 0)
        preds = np.int64(predictions > 0)

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        ############################################
        predictions = np.dot(X ,w) + b
        predictions = sigmoid(predictions)
        preds = np.int64(predictions >= 0.5)


    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds
#softmax implementation
def softmax(x):
    expoenent_x = np.exp(x - np.max(x,axis=0))
    return expoenent_x / np.sum(expoenent_x, axis= 0)

def softmaxWithoutOverflow(x):
    expoenent_x = np.exp(x)
    return expoenent_x / np.sum(expoenent_x, axis= 0)

def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        # weight bias matrix
        wb = np.column_stack((w, b))
        # adding for bias
        X = np.column_stack((X, np.array([1] * N)))
        for i in range(max_iterations):
            random_index = np.random.choice(N)
            X_sample = X[random_index]
            Y_sample = y[random_index]
            wTx = np.dot(wb, X_sample.transpose())
            softmax_result = softmax(wTx)
            #update the class with -1, y == yn
            softmax_result[Y_sample] = softmax_result[Y_sample] - 1

            #we need a (c) * (d + 1) dimensions
            #change softmaxresult to (c * 1)
            softmax_result = np.reshape(softmax_result, (C, 1))

            #change the xn to 1 * d + 1
            X_sample = np.reshape(X_sample,(1,D+1))
            result = np.dot(softmax_result,X_sample)
            wb = wb - (step_size*result)



    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        ############################################
        # weight bias matrix
        wb = np.column_stack((w, b))
        # adding for bias
        X = np.column_stack((X, np.array([1] * N)))

        #using one hot for easier computation
        y_one_hot = np.zeros((N, C))
        y_one_hot[np.arange(N), y] = 1

        for i in range(max_iterations):
            wTx = np.dot(wb, X.transpose())
            softmax_result = softmax(wTx)
            # update the class with -1, y == yn
            softmax_result = softmax_result - y_one_hot.transpose()

            result = np.dot(softmax_result, X)
            #average gradient
            wb = wb - ((step_size/N) * result)

    else:
        raise "Type of Gradient Descent is undefined."
    
    #weights can be got from here
    w = wb[:,:D]
    b = wb[:,D]
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    ############################################
    #make determistic prediction
    xTw = np.dot(X, w.transpose())
    xTwb = xTw + b
    #find the maximum along the row(axis = 1)
    preds = xTwb.argmax(axis=1)

    assert preds.shape == (N,)
    return preds