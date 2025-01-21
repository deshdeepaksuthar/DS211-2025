# Naming should be Lec02_20250116.py
import numpy as np
import pandas as pd

df = pd.read_csv("Lec02-20250116/dataset/real_estate_dataset.csv")

n_samples, n_features = df.shape

columns = df.columns

np.savetxt("column_names.txt", columns, "%s")

# we will only use the first 4 or 5 columns not use all 12 of them,:
#     id, Square_Foot, Garage_Size,  Location_Score, Distance_to_Center as features
#Cols: 1, 2, 6, 7

X = df[["Square_Feet", "Garage_Size", "Location_Score", "Distance_to_Center"]]

y = df["Price"].values
print(f"Length of dataset:{X.shape}" )

n_samples, n_features = X.shape

# making a linear model to predict the prices
# to do that we need an array of size n_features +1  and initialize to 1
# plus one is for bias, the mean value

coeffs = np.ones(n_features+1)

# predict the price of each sample in X
predictions_by_defs = X @ coeffs[1:] + coeffs[0]

### TESTING
# print("TEST", np.ones((n_samples+1)).shape)

# append a stack of 1s to X
X = np.hstack((np.ones((n_samples,1)), X))

predictions = X @ coeffs

# one can check if they are the same
errors = y - predictions

# calculate the mean of square of errors using a loop (Brute force)
loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i]**2
loss_loop = loss_loop/n_samples

# calculate the errors using Matrix operations
loss_matrix = np.transpose(errors)@errors /n_samples

# calculating the errors for all ones set of coefficients
errors = y - predictions

rel_errors = errors / y
print(f"Errors size: {errors.size}")
print(f"Errors size: {np.linalg.norm(errors)}")
print(f"Errors size: {np.linalg.norm(rel_errors)}")

# what is my optimization problem
# arrive at the coefficeients the minimize the least square error
# this is the least squares problem

# objective function: ~~~ Sum of errors
# the solution has certain properties like
# the gradient at that point is 0
# and the hessian is positive at that point

# we have to search for one such points where grad f is zero
# or i can set a generalized funciton to 0 and then look for the point, grad f delta r = 0

# loss-matrix in terms of data
loss_matrix = (y-X@coeffs).T @ (y - X@coeffs) /n_samples

# calculate the gradient matrix
grad_matrix = -2/n_samples *X.T @( y - X@coeffs)

# we set grad_matrix = 0 and find the solutions(coeffs)

# X.T @ y = X.T @ X @ coeffs
# X.T @ X @ coeffs = X.T @ y // This is the normal equation
coeffs = np.linalg.inv(X.T @ X ) @ X.T @ y
np.savetxt("coeffs.csv", coeffs, delimiter=",")

prediction_model = y -predictions
rel_errors_model = errors/y
print(f"L2 norm of errors_model: {np.linalg.norm(rel_errors_model)}")

# some of it is done, now we will use all the features of the dataset

X = df.drop("Price", axis=1).values

y = df["Price"].values

n_samples, n_features = X.shape
X = np.hstack((np.ones((n_samples,1)), X))
coeffs = np.linalg.inv(X.T @ X ) @ X.T @ y

np.savetxt("coeffsall.csv", coeffs, delimiter=",")

rank_XTX = np.linalg.matrix_rank(X.T @ X)



# The inversion process may not work all the time,
# so we do matrix QR decomposition

Q, R = np.linalg.qr(X)

# X = Q R
# R *coeffs = b

b = Q.T @ y
# one can do this
## coeffs_qr = np.linalg.inv(R) @ b
# but we dont want inversion
# doing a loop to do the back subtituition

coeffs_qr_loop = np.zeros(n_features+1)

for i in range(n_features, -1 , -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coeffs_qr_loop[i] = coeffs_qr_loop[i] - R[i,j]*coeffs_qr_loop[j]
    coeffs_qr_loop[i] = coeffs_qr_loop[i]/R[i,i]

np.savetxt("coeffs_qr_loop.csv", coeffs_qr_loop, delimiter=",")

# Solving using SVD method
U, S, Vt = np.linalg.svd(X, full_matrices=False)

coeffs_svd = Vt.T @ np.diag(1/S) @ U.T @ y
coeffs_svd_pinv = np.linalg.pinv(X) @ y
np.savetxt("coeffs_svd.csv", coeffs_svd, delimiter=",")
np.savetxt("coeffs_svd_pinv.csv", coeffs_svd_pinv, delimiter=",")

import matplotlib.pyplot as plt
X_feature = np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.01)
plt.plot(X[:, 2], y, 'o')
# plt.plot(X[:, 1], X[:, 0:2] @ coeffs[0:2], color='red')
# plt.plot(X_feature, X_feature * coeffs_svd[1] , color='magenta')
plt.xlabel("Area, Sq Ft")
plt.ylabel("Prices")
plt.title("Prices vs area")
## try again
# X_1 = X[:, 2]
# coeffs_1 = np.linalg.inv(X_1.T @ X_1) @X_1.T @ y
# X_feature = np.arange(np.min(X[:,0]), np.max(X[:,0]), 0.01)
# plt.plot(X_feature, X_feature * coeffs_1[1])

# X[2] is the area not X[1]

X = df["Square_Feet"].values
y = df["Price"].values
X = np.hstack((np.ones((n_samples,1)), X.reshape(-1,1)))
coeffs_1 = np.linalg.inv(X.T @ X ) @ X.T @ y
print(coeffs_1)
plt.plot(X_feature, X_feature * coeffs_1[1] + coeffs_1[0])
plt.show()
