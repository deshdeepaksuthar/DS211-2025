import numpy as np
import matplotlib.pyplot as plt
from datamaker import RegressionDataMaker

# change noise to 0.1 and samples to 100 if this works
data_maker = RegressionDataMaker(n_samples=100, n_features=1,noise=0.1, seed=42)

X,y,coefs = data_maker.make_data_with_ones()

data_maker.save_data_csv(X,y, "data.csv")

data_maker.save_coefs_csv(coefs, "true_coefs.csv")
def linear_model(X, theta):
    return X@ theta
# make a least squares objective finction for regression
def mse_linear_regression(X,y,theta):
    n_samples = X.shape[0]
    err = linear_model(X,theta)-y
    sum_sq_err = 0
    for i in range(n_samples):
        sum_sq_err = sum_sq_err +  err[i] ** 2

    return (1/n_samples) * sum_sq_err
def gradient_mse_linear_regression(X,y, theta):
    n_samples = X.shape[0]
    return (2/n_samples) * X.T @(X@ theta - y)


step_length = 0.1
n_iterations = 100
#theta_0 = np.ones((X.shape[1], 1))
theta_0 = np.array([[2], [2]])
# print the number of features
# print shape of theta_0
def gradient_descent(X, y, mse_linear_regression, gradient_mse_linear_regression, step_length, n_iterations, theta_0, tol=1e-6):
    n_samples , n_features = X.shape
    theta = theta_0
    path = theta
    iter_count = 0
    while np.linalg.norm(gradient_mse_linear_regression(X,y,theta)) > tol :
        theta = theta - step_length * gradient_mse_linear_regression(X,y, theta)
        path = np.hstack((path, theta))
        iter_count += 1
        if iter_count > n_iterations:
            break
        # if iter_count%10 == 0:
        #    print(f"Iteration {i} , MSE: {mse_linear_regression(X,y, theta)}, theta={theta.faltten()}")
    return theta, path

def plot_contour_with_path(X, y, mse_linear_regression, path, theta):
    theta0_vals = np.linspace(-5,5,10)
    theta1_vals = np.linspace(-5,5,10)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));
    for i, theta_0 in enumerate(theta0_vals):
        for j, theta_1 in enumerate(theta1_vals):
            J_vals[i,j] = mse_linear_regression(X,y,np.array([[theta_0], [theta_1]]));
    plt.contourf(theta0_vals, theta1_vals, J_vals.T)
    # plt.contourf(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2,3,20))
    plt.plot(path[0], path[1], marker='x', color='black')
    for i in range(path.shape[1]):
        plt.text(path[0,i], path[1,i], str(i), color='black')
    plt.plot(theta[0], theta[1], marker='x', color='red')
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.savefig(f"contour_mse_with_steplength{step_length}.png")
    plt.show()

theta, path = gradient_descent(X,y,mse_linear_regression, gradient_mse_linear_regression, step_length, n_iterations, theta_0)
plot_contour_with_path(X,y, mse_linear_regression, path, theta)


def plot_contour(X, y, mse_linear_regression , id):
    theta0_vals = np.linspace(-5,5,10)
    theta1_vals = np.linspace(-5,5,10)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));
    for i, theta_0 in enumerate(theta0_vals):
        for j, theta_1 in enumerate(theta1_vals):
            J_vals[i,j] = mse_linear_regression(X,y,np.array([[theta_0], [theta_1]]));
    # plt.contourf(theta0_vals, theta1_vals, J_vals.T)
    # plt.contourf(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2,3,20))
    plt.contourf(theta0_vals, theta1_vals, J_vals.T, levels=np.arange(0,100,10))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.savefig(f"contour_mse_{id}.png")
    plt.show()

plot_contour(X[0:3,:], y[0:3], mse_linear_regression, "oto3")
plot_contour(X[0:3,:], y[3:6], mse_linear_regression, "3to6")
