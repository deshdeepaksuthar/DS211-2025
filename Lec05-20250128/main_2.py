# save it as name_v2
import numpy as np
import matplotlib.pyplot as plt

# weight of people
y = np.array([105,74,63])
# y = y.reshape(-1,1)
x = np.array([182, 175, 170])
# x = x.reshape((1,-1))

X = np.vstack((np.ones(3),x)).T

coefs = np.array([0,0])
coefs = coefs.reshape((1,-1))


def f(coefs, X, y):
    y_pred =  X@coefs
    return np.sum((y-y_pred)**2)

def grad_f(coefs, X, y):
    y_pred = X @ coefs
    return -2*X.T@(y - y_pred)

def gradient_descent_nobacktracking(f, grad_f, x_0, X, y, step_length=0.1,  max_iter=1000, tol=1e-6):
    x = x_0
    path = x_0
    for ii in range(max_iter):
        if np.linalg.norm(grad_f(x, X, y)) < tol:
            break
        dir_descent = - grad_f(x, X, y)
        x = x + step_length * dir_descent
        # save the path of descent
        path = np.vstack((path, x))
    return x, f(x, X, y), ii, path
step_length = 1.2
x_optim, f_x, num_iter, path = gradient_descent_nobacktracking(f, grad_f, coefs, X, y,  step_length)
print("Numer of iterations", num_iter)
print("minimum point",x )




xx = np.linspace(-10,10,100)
yy = np.linspace(-10,10,100)

XX , YY = np.meshgrid(xx, yy)
ZZ = np.zeros((100,100))
for ii in range(100):
    for jj in range(100):
        ZZ[ii,jj] = f([X[ii,jj], Y[ii,jj]], X,y)

cfp = plt.contourf(XX,YY,ZZ, levels=np.linspace(0,100,10), cmap='Blues', extend='max', vmin=0, vmax=100)
plt.colorbar(cfp)
plt.plot(path[:,0], path[:,1], marker='o')

plt.plot(coefs[0], coefs[1], marker='o', color='green')
plt.plot(x[0], x[1], marker='o', color='red')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title(f"Contour plot of f(X) with direction descent with no backtracking step_length{step_length}")
for ii in range(path.shape[0]):
    plt.text(path[ii,0], path[ii,1], str(ii))
plt.show()
plt.savefig(f"f_contour_with_dnb_ls_alpha{alpha}_rho{rho}_no_backtracking.png")
