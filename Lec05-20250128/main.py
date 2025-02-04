import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0]*x[1]

def grad_f(x):
    return np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])

# backtracking line search
def backtracking_line_search(f, grad_f, x, dir_descent, alpha=1, rho=0.8, c=1e-4):
    # dir_descent = p from nocedal page 36 or 37
    # beta is rho
    while ( f(x + alpha * dir_descent) > f(x) + c*alpha* grad_f(x).T @ dir_descent):
        alpha = rho*alpha
    return alpha

# a method to define if a given direction is a valid descent direction
def is_descent_direction(grad_f, x, dir_descent):
    return grad_f(x).T @ dir_descent < 0

def gradient_descent(f, grad_f, x_0, alpha, rho,  max_iter=1000, tol=1e-6):
    x = x_0
    path = x_0
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        dir_descent = - grad_f(x)
        step_length = backtracking_line_search(f, grad_f, x, dir_descent, alpha, rho)
        x = x + step_length * dir_descent
        # save the path of descent
        path = np.vstack((path, x))

    return x, f(x), ii, path

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

X , Y = np.meshgrid(x,y)
Z = f([X,Y])

cfp = plt.contourf(X,Y,Z, levels=np.linspace(0,100,10), cmap='Blues', extend='max', vmin=0, vmax=100)
plt.colorbar(cfp)
# plt.show()
plt.savefig("f_contour.png")

alpha = 1
rho = 0.8
x_0 = np.array([5,5])
x, f_x, num_iter, path = gradient_descent(f, grad_f, x_0, alpha, rho)
print("Numer of iterations", num_iter)
print("minimum point",x[0], x[1] )

plt.plot(path[:,0], path[:,1], marker='o')

plt.plot(x_0[0], x_0[1], marker='o', color='green')
plt.plot(x[0], x[1], marker='o', color='red')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Contour plot of f(X) with direction descent")
for ii in range(path.shape[0]):
    plt.text(path[ii,0], path[ii,1], str(ii))
plt.show()
plt.savefig(f"f_contour_with_descent_alpha{alpha}_rho{rho}.png")
plt.close()

def gradient_descent_nobacktracking(f, grad_f, x_0, step_length=0.1,  max_iter=1000, tol=1e-6):
    x = x_0
    path = x_0
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        dir_descent = - grad_f(x)
        x = x + step_length * dir_descent
        # save the path of descent
        path = np.vstack((path, x))

    return x, f(x), ii, path
x_0 = np.array([5,5])
step_length = 0.1
x, f_x, num_iter, path = gradient_descent_nobacktracking(f, grad_f, x_0, step_length)

print("Numer of iterations", num_iter)
print("minimum point",x )
cfp = plt.contourf(X,Y,Z, levels=np.linspace(0,100,10), cmap='Blues', extend='max', vmin=0, vmax=100)
plt.colorbar(cfp)
plt.plot(path[:,0], path[:,1], marker='o')

plt.plot(x_0[0], x_0[1], marker='o', color='green')
plt.plot(x[0], x[1], marker='o', color='red')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title(f"Contour plot of f(X) with direction descent with no backtracking step_length{step_length}")
for ii in range(path.shape[0]):
    plt.text(path[ii,0], path[ii,1], str(ii))
plt.show()
plt.savefig(f"f_contour_with_descent_alpha{alpha}_rho{rho}_no_backtracking.png")
