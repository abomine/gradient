import numpy as np
from sklearn.metrics import mean_squared_error

'''

Here's the algorithm for gradient descent that i will implement:

1/ Initialize the parameters of the model with predefined values [w ~ weight is 0.1 here & b ~ bias is 0.01].
2/ Set the learning rate for the algorithm [0.0001 here].
3/Repeat until convergence or a maximum number of iterations is reached:
    Compute the predicted values [y_predicted] of the model using the current parameter values.
    Calculate the gradient of the cost function with respect to the parameters.
    Update the parameters by subtracting the learning rate times the gradient from the current parameter values.
    Track and record the cost function or other metrics to monitor the optimization process.
4/Return the final parameter values obtained after reaching the maximum number of iterations.

'''


#gradient_descent_function

def gradient_descent(X, Y, n_iter=1000, learning_rate=0.001, stop_threshold=1e-6):
    #Init
    current_weight = 0.1 
    current_bias = 0.01
    n = float(len(X))

    costs = []
    weights = []
    previous_cost = None

    for i in range(n_iter):
        #calculating predictions
        y_predicted = (current_weight*X) + current_weight

        #calculating new cost
        current_cost = mean_squared_error(Y, y_predicted)

        if previous_cost and abs(previous_cost-current_cost)<=stop_threshold:
            break

        #Updating
        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight)
         
        #calculating gradients
        weight_derivative = -(2/n) * sum(X * (Y-y_predicted))
        bias_derivative = -(2/n) * sum(Y-y_predicted)
         
        # Updating parameters
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")




