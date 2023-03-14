import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from tqdm import tqdm


def hw2q2():
    Ntrain = 100
    data, labels = generateData(Ntrain)
    plot3(data[0, :], data[1, :], data[2, :], labels[0], 'Training')
    xTrain = data[0:2, :]
    yTrain = data[2, :]

    Ntrain = 1000
    data, labels = generateData(Ntrain)
    # plot3(data[0, :], data[1, :], data[2, :], labels[0], 'Validation')
    xValidate = data[0:2, :]
    yValidate = data[2, :]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x, labels


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        x[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    return x, labels


def plot3(a, b, c, labels=None, kind='Training', col="b"):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    # markers = ['o', 'x', '^']
    # for i in range(labels.shape[0]):
    #     ax.scatter(a[i], b[i], c[i], marker=markers[int(labels[i])-1], color=col)
    ax.scatter(a, b, c, marker='o', color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(kind + ' Dataset')
    plt.show()


def mse(estimate, true):
    return np.mean((estimate - true) ** 2)


def mle_estimate(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def map_estimate(X, y, gamma):
    return np.linalg.inv(X.T.dot(X) + (1 / gamma)*np.eye(X.shape[1])).dot(X.T).dot(y)


def map_classification():
    gammas = np.geomspace(10e-5, 10e5, num=1000)
    mse_ = np.zeros(1000)
    best_w = None
    min_mse = 10
    for i, gamma in enumerate(gammas):
        w = map_estimate(train_x_quad, train_y, gamma)
        pred_y = X_valid_cubic.dot(w)
        mse_[i] = mse(pred_y, validate_y)
        if min_mse > mse_[i]:
            min_mse = mse_[i]
            best_w = w

    pred_y = X_valid_cubic.dot(best_w)
    plot3(validate_x[0, :], validate_x[1, :], validate_y, validate_y, 'Validation')
    plot3(validate_x[0, :], validate_x[1, :], pred_y, pred_y, 'MAP Prediction')

    _, ax = plt.subplots(figsize=(10, 10))
    ax.plot(gammas, mse_, color='b')

    ax.set_xscale('log')
    ax.set_xticks(np.geomspace(10e-5, 10e5, num=15))
    ax.set_xlabel("gamma")
    ax.set_ylabel("MSE")
    ax.set_title("Mean sqaure error for MAP Estimator for different gamma")
    plt.show()
    return gammas[np.argmin(mse_)], np.min(mse_)


if __name__ == '__main__':
    train_x, train_y, validate_x, validate_y = hw2q2()
    phi = PolynomialFeatures(degree=3)
    train_x_quad = phi.fit_transform(train_x.T)

    theta_mle = mle_estimate(train_x_quad, train_y)

    X_valid_cubic = phi.transform(validate_x.T)
    y_pred_mle = X_valid_cubic.dot(theta_mle)

    mse_mle = mse(y_pred_mle, validate_y)

    print('MSE for MLE', mse_mle)
    plot3(validate_x[0, :], validate_x[1, :], validate_y, validate_y, 'Validation')
    plot3(validate_x[0, :], validate_x[1, :], y_pred_mle, y_pred_mle, 'MLE Prediction')

    gamma_, mse_ = map_classification()
    print(r"MSE for MAP with gamma={} is: {:.3f}".format(gamma_, mse_))

