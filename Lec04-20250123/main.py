import numpy as np

# making a 1d scalar function

def f(x):
    return 2*x**2 + 3*x + 4

# visualising this function
x = np.linspace(-10,10,100)
import matplotlib.pyplot as plt
plt.plot(x, f(x))
plt.savefig("f_x.png")
plt.close()

# define the derivative of  f
def grad_f(x):
    return 4*x + 3

# make an initial guess of x optimum and go on from there
x_0 = 5

print(f"f(x_0): {f(x_0)}")
print(f"grad_f(x_0): {grad_f(x_0)}")

dir_descent = - grad_f(x_0)

# let us consider a step length that defines the amount of movement that I will perfoem in the direction of the descent

#x_new = x_0 + step_lengeth * dir_descent

# what is the function value of x_new
# total_movement = step_length * dir_descent
# f_x_new = f(x_0 + step_length * dir_descent)
# expand f in taylor series
# and then use the first order approximation
# f_x_new = f(x_0) + grad_f(x_0) * step_length * dir_descent + ...
# f_x_new = f(x_0) + grad_f(x_0) * total_movement + ...
# f_x_new (step_length) = f(x_0) + gradf(x_0) * step_length *dir_descent

# Goal: find step_length such that f_x_new(step_length) is minimized
# step_length sohuld be decided such that there is sufficient decrease
# in 1-dim the problem of finding direction is trivial
# the second order approximation is
# f_x_new (step_length) = f(x_0) + gradf(x_0) * step_length *dir_descent + 0.5* step_length**2 * dir_descent^T * hessian_f(x_0) * dir_descent

# consider in 2-dim
def f2(x1, x2):
    return x1**2 + x2**2 + 0.5*x1*x2
def grad_f2(x1, x2):
    return np.array([2*x1 + 0.5*x2, 2*x2 + 0.5*x1])

x1 = np.linspace(-4,4,100)
x2 = np.linspace(-4,4,100)
[X1, X2] = np.meshgrid(x1,x2)

Y = f2(X1, X2)
plt.set_cmap("viridis")
plt.contourf(X1, X2, Y, 200)
plt.colorbar()
plt.clim(0,10)
# plt.show()
plt.savefig("f2_x1_x2_contour.png")

# a funciton to find the minimum of a 1D scalar funciton
dir_descent = - grad_f2(x1,x2) # direction of steepest descent
new_x1 = x1 + step_length * dir_descent[0];
new_x2 = x2 + step_length * dir_descent[1];

# steps:
# 1. python method to find the minimum of a 1D scalar function
# 2. use that together with the gradient of the 2D scalar function to code the iterative gradient descent
