import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def generate_data(mu, sigma, label_name):
    x = np.random.normal(mu, sigma, n)
    y = np.random.normal(mu, sigma, n)
    label = np.array([label_name for i in range(n)])
    return np.vstack((x, y)).T, label

def scatter_plot(x, y, c, title):
    plt.scatter(x, y, c=c)
    plt.title(title)

class LogiscticRegression():
    def __init__(self, mode='newton', alpha=0.1, maxiter=10, threshold=0.0005):
        self.mode = mode.lower()
        self.alpha = 0.1
        self.maxiter = maxiter
        self.threshold = threshold
    
    def sigmoid_func(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_func(self, X, y, p):
        return np.dot(X.T, (y - p))

    def hessian_func(self, X, p):
        W = np.eye(X.shape[0]) * (p * (1 - p))
        return np.dot(np.dot(X.T, W), X).T
    
    def cost_func(self, z, y):
        p = self.sigmoid_func(z)
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
    
    def predict_prob(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return self.sigmoid_func(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()
    
    def confusion_matrix(self, X, y):
        y_pred = self.predict(X).flatten()
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        tn = np.sum((y_pred == 0) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        return tp, fp, tn, fn
    
    def gradient_descent(self, X, y, z, alpha):
        p = self.sigmoid_func(z)
        gradient = self.gradient_func(X, y, p) / X.shape[0]
        self.theta = self.theta + alpha * gradient

    def newton(self, X, y, z):
        p = self.sigmoid_func(z)
        gradient = self.gradient_func(X, y, p)
        hessian = self.hessian_func(X, p)

        self.theta = self.theta + np.dot(np.linalg.inv(hessian), gradient)
        
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        y = np.expand_dims(y, axis=1)
        self.n, self.m = X.shape
        
        self.theta = np.zeros((self.m, 1))
        z = np.dot(X, self.theta)
        self.loss = self.cost_func(z, y)
        # print(self.loss)
        
        while True:
            if self.mode == 'newton':
                self.mode_title = "Newton's method"
                self.newton(X, y, z)
            elif self.mode == 'gradient':
                self.mode_title = "Gradient descent"
                self.gradient_descent(X, y, z, self.alpha)
            z = np.dot(X, self.theta)
            
            new_loss = self.cost_func(z, y)
            # print(new_loss)
            error = new_loss - self.loss
            if abs(error) < self.threshold:
                break
            else:
                self.loss = new_loss
                
    def show_table(self, X, y):
        print('\n%s:'%self.mode_title)
        print('\nw:')
        for theta in self.theta:
            print('%12f'%theta[0])
        
        tp, fp, tn, fn = self.confusion_matrix(X, y)
        print('\nConfusion Matrix:')
        print('\t\t%20s%20s'%('Predict cluster 1', 'Predict cluster 0'))
        print('Is cluster 1 %20s %20s'%(tp, fp))
        print('Is cluster 0 %20s %20s\n'%(fn, tn))
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print('Sensitivity (Successfully predict cluster 1): ', sensitivity)
        print('Specificity (Successfully predict cluster 2): ', specificity)


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--random_seed", help="set the random seed.", default=1, type=int)
    parser.add_argument("--n", help="sample size of each class.", default=50, type=int)
    parser.add_argument("--mu", help="the mu of class 0 and class 1", default='1,3')
    parser.add_argument("--var", help="the variance of class 0 and class 1", default='2,4')
    parser.add_argument("--mode", help="use Gradient Descent or Newton to optimize", default='both')
    args = parser.parse_args()

    np.random.seed(0)
    n = args.n
    D1_mu, D2_mu = tuple([int(i) for i in args.mu.split(',')])
    D1_var, D2_var = tuple([int(i) for i in args.var.split(',')])
    mode = args.mode

    D1 = generate_data(mu=D1_mu, sigma=D1_var, label_name=0)
    D2 = generate_data(mu=D2_mu, sigma=D2_var, label_name=1)
    X = np.vstack((D1[0], D2[0]))
    y = np.hstack((D1[1], D2[1]))

    if mode == 'both':
        n_mode = 30
    elif mode in ['newton', 'gradient descent']:
        n_mode = 20
    else:
        print("'please the mode as gaussian and newton'")
    plt.figure(figsize=(10,7))
    plt.subplot(100+n_mode+1)
    scatter_plot(X[:,0], X[:,1], c=y, title='Ground truth')

    pred_list = []
    mode_list = []
    if mode in ['gradient', 'both']:
        lr_gradient = LogiscticRegression(mode='gradient')
        lr_gradient.fit(X, y)
        lr_gradient.show_table(X, y)
        y_gradient = lr_gradient.predict(X)
        pred_list.append(y_gradient)
        mode_list.append(lr_gradient.mode_title)

    if mode in ['newton', 'both']:
        lr_newton = LogiscticRegression(mode='newton')
        lr_newton.fit(X, y)
        lr_newton.show_table(X, y)
        y_newton = lr_newton.predict(X)
        pred_list.append(y_newton)
        mode_list.append(lr_newton.mode_title)

    for i in range(len(pred_list)):
        plt.subplot(100+n_mode+(i+2))
        scatter_plot(X[:,0], X[:,1], c=pred_list[i], title=mode_list[i])
    plt.show()
