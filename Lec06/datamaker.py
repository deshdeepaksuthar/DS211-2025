import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class RegressionDataMaker:
    def __init__(self, n_samples=100, n_features=1, noise=0.1, seed=42, true_coef=np.array([[1], [2]])):
        self.n_samples = n_samples
        self.n_features = n_features
        self.seed = seed
        self.noise = noise;
        self.true_coef = true_coef

    def make_data(self):
        X, y, true_coefs = make_regression(n_samples=self.n_samples, n_features=self.n_features,
                                      noise=self.noise, coef=True, random_state=self.seed)
        coefs = self.true_coef
        y = X@coefs[1:]  + coefs[0] + np.random.normal(0, self.noise, self.n_samples).reshape(-1,1)
        return X,y, coefs

    #save the coefs to a csv
    def save_coefs_csv(self, coefs, filename):
        np.savetxt(filename, coefs, delimiter=',')
        print(f"Data saved in {filename}")

    #save the data
    def save_data_csv(self, X,y, filename):
        np.savetxt(filename, np.column_stack((X,y)), delimiter=',')
        print(f"Data saved in {filename}")

    def plot_data(self, X,y):
        plt.scatter(X,y, color='black')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.show()


    def make_data_with_ones(self):
        X, y, coefs = self.make_data()
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X, y, coefs
