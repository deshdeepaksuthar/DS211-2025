# Naming should be Lec02_20250116.py
import numpy as np
import pandas as pd

df = pd.read_csv("dataset/real_estate_dataset.csv")

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

coefs = np.ones(n_features+1)

# predict the price of each sample in X
predictions_by_defs = X @ coefs[1:] + coefs[0]

### TESTING
# print("TEST", np.ones((n_samples+1)).shape)

# append a stack of 1s to X
X = np.hstack((np.ones((n_samples,1)), X))

predictions = X @ coefs

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
# print(f"Errors size: {errors.size}")
# print(f"Errors size: {np.linalg.norm(errors)}")
# print(f"Errors size: {np.linalg.norm(rel_errors)}")

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
loss_matrix = (y-X@coefs).T @ (y - X@coefs) /n_samples

# calculate the gradient matrix
grad_matrix = -2/n_samples *X.T @( y - X@coefs)

# we set grad_matrix = 0 and find the solutions(coefs)

# X.T @ y = X.T @ X @ coefs
# X.T @ X @ coefs = X.T @ y // This is the normal equation
print("Computing using direct matrix inversion")
coefs = np.linalg.inv(X.T @ X ) @ X.T @ y
np.savetxt("results/coefs.csv", coefs, delimiter=",")
print("Output saved to: results/coefs.csv")

prediction_model = y -predictions
rel_errors_model = errors/y

# some of it is done, now we will use all the features of the dataset

X = df.drop("Price", axis=1).values

y = df["Price"].values

print("Computing using direct matrix inversion for all 11 features")
n_samples, n_features = X.shape
X = np.hstack((np.ones((n_samples,1)), X))
coefs = np.linalg.inv(X.T @ X ) @ X.T @ y

np.savetxt("results/coefsall.csv", coefs, delimiter=",")
print("Output saved to: results/coefsall.csv")

# rank_XTX = np.linalg.matrix_rank(X.T @ X)



# The inversion process may not work all the time,
# so we do matrix QR decomposition

Q, R = np.linalg.qr(X)

# X = Q R
# R *coefs = b

b = Q.T @ y
# one can do this
## coefs_qr = np.linalg.inv(R) @ b
# but we dont want inversion
# doing a loop to do the back subtituition

print("Computing using QR decomposition for all 11 features")
coefs_qr_loop = np.zeros(n_features+1)

for i in range(n_features, -1 , -1):
    coefs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coefs_qr_loop[i] = coefs_qr_loop[i] - R[i,j]*coefs_qr_loop[j]
    coefs_qr_loop[i] = coefs_qr_loop[i]/R[i,i]

np.savetxt("results/coefs_qr_loop.csv", coefs_qr_loop, delimiter=",")
print("Output saved to: results/coefs_qr_loop.csv")

# Solving using SVD method

# X = U S V^T
# eigen value decomposition
# A = V D V^T
# A^-1 = V D^-1 V^T
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# we are finding the pseudo inverse of X, (the least square sense inverse)
# X inv ~= X dagger
# X dagger = (X^T X )^-1 X^T
# Homework part
# Why not just
# print(np.linalg.pinv(X) @ y)

# normal equation is Xt X coef = Xt y
# coef = V ( S.T S ) ^-1 S.T  U.T y

print("Computing using SVD for all 11 features")
S_inv = 1/(S*S);
S_inv = np.diag(S_inv);
S = np.diag(S)
padded_S = np.diag(S_inv)
coefs_svd = Vt.T @ S_inv @ S @ U.T @ y ;

np.savetxt("results/coefs_svd.csv", coefs_svd, delimiter=",")
print("Output saved to: results/coefs_svd.csv")
