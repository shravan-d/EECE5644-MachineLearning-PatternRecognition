import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from tqdm import tqdm


# Generate samples
def gen_samples(count, generate_new=False, kind='Train'):
    if generate_new:
        samples_L0_w01 = multivariate_normal.rvs(mean=m01, cov=C, size=int(count * 0.5 * p0))
        samples_L0_w02 = multivariate_normal.rvs(mean=m02, cov=C, size=int(count * 0.5 * p0))
        samples_L1_w11 = multivariate_normal.rvs(mean=m11, cov=C, size=int(count * 0.5 * p1))
        samples_L1_w12 = multivariate_normal.rvs(mean=m12, cov=C, size=int(count * 0.5 * p1))
        samples = np.concatenate((samples_L0_w01, samples_L0_w02, samples_L1_w11, samples_L1_w12), axis=0)
        labels = [0] * int(count * p0) + [1] * int(count * p1)
        np.save('data/'+kind + 'Samples' + str(count), samples)
        np.save('data/'+kind + 'Labels' + str(count), labels)
        return np.array(samples), np.array(labels)
    else:
        samples = np.load('data/'+kind + 'Samples' + str(count) + '.npy')
        labels = np.load('data/'+kind + 'Labels' + str(count) + '.npy')
        return samples, labels


def classify(N, samples, true_labels):
    label1_count = sum(true_labels)
    label0_count = N - label1_count

    # Class-conditional Gaussian distributions
    gaussian_01 = multivariate_normal(mean=m01, cov=C)
    gaussian_02 = multivariate_normal(mean=m02, cov=C)
    gaussian_11 = multivariate_normal(mean=m11, cov=C)
    gaussian_12 = multivariate_normal(mean=m12, cov=C)

    # Compute likelihood ratio for each sample
    L = np.zeros(N)
    for i in range(N):
        L[i] = ((gaussian_11.pdf(samples[i]) + gaussian_12.pdf(samples[i])) * p1) \
               / ((gaussian_01.pdf(samples[i]) + gaussian_02.pdf(samples[i])) * p0)

    true_positive_rates, false_positive_rates = [], []
    min_error_gamma = -1
    min_error, min_theoretical_error = 10000000, -1
    min_true_positive_rate, min_false_positive_rate = -1, -1
    precision = 0.001

    for gamma_threshold in tqdm(range(-10000, 10000)):
        predicted_labels = np.zeros(N)
        predicted_labels[L > precision * gamma_threshold] = 1

        # Calculate metrics
        TP = np.sum(np.logical_and(predicted_labels == 1, true_labels == 1))
        TN = np.sum(np.logical_and(predicted_labels == 0, true_labels == 0))
        FP = np.sum(np.logical_and(predicted_labels == 1, true_labels == 0))
        FN = np.sum(np.logical_and(predicted_labels == 0, true_labels == 1))
        true_positive_rates.append(TP / float(label1_count))
        false_positive_rates.append(FP / float(label0_count))
        error = (FP / float(label0_count)) * p0 + (FN / float(label1_count)) * p1

        if precision * gamma_threshold == p0 / p1:
            min_theoretical_error = error
        if error < min_error:
            min_error = error
            min_error_gamma = precision * gamma_threshold
            min_true_positive_rate, min_false_positive_rate = TP / float(label1_count), FP / float(label0_count)

    print('Experimental minimum error: {}\nExperimental minimum gamma: {}\n'
          'Theoretical minimum error: {}\nTheoretical minimum gamma: {}'
          .format(min_error, min_error_gamma, min_theoretical_error, p0 / p1))

    return true_positive_rates, false_positive_rates, min_true_positive_rate, min_false_positive_rate


def sigmoid(x): return 1 / (1 + np.exp(-x))


def negative_log_likelihood(X, w, label):
    z = np.dot(X, w)
    cost0 = label.T.dot(np.log(sigmoid(z)))
    cost1 = (1 - label).T.dot(np.log(1 - sigmoid(z)))
    cost = -(cost1 + cost0) / len(label)
    return cost


def train_logistic_linear_function(X, y, alpha=0.01, iterations=1000, quadratic_function=False):
    if quadratic_function:
        quadratic = PolynomialFeatures(degree=2)
        X = quadratic.fit_transform(X)
    w = np.zeros(X.shape[1])
    cost_list = np.zeros(iterations, )
    for i in range(iterations):
        w = w - alpha * np.dot(X.T, sigmoid(np.dot(X, w)) - y)
        cost_list[i] = negative_log_likelihood(X, w, y)
    # print('Initial Cost: {:.3f}, Final Cost: {:.3f}'.format(cost_list[0], cost_list[-1]))
    return w


def predict(X, w, true_labels, quadratic_function=False):
    true_positive_rates, false_positive_rates = [], []
    label1_count = sum(true_labels)
    label0_count = X.shape[0] - label1_count
    if quadratic_function:
        quadratic = PolynomialFeatures(degree=2)
        z = np.dot(quadratic.fit_transform(X), w)
    else:
        z = np.dot(X, w)
    predicted_labels = sigmoid(z)
    predicted_labels[predicted_labels > 0.5] = 1
    predicted_labels[predicted_labels < 1] = 0
    TP = np.sum(np.logical_and(predicted_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(predicted_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(predicted_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(predicted_labels == 0, true_labels == 1))
    true_positive_rates.append(TP / float(label1_count))
    false_positive_rates.append(FP / float(label0_count))
    error = (FP / float(label0_count)) * p0 + (FN / float(label1_count)) * p1
    print('Probability of Error: ', error)
    for i in range(X.shape[0]):
        if true_labels[i] == 0 and predicted_labels[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], marker='x', color='green')
        if true_labels[i] == 0 and predicted_labels[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], marker='x', color='red')
        if true_labels[i] == 1 and predicted_labels[i] == 0:
            plt.scatter(X[i, 0], X[i, 1], marker='o', color='red')
        if true_labels[i] == 1 and predicted_labels[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], marker='o', color='green')

    if quadratic_function:
        def linear_quadratic_func(x, w):
            return w[0] + w[1] * x[:, 0] + w[2] * x[:, 1] + w[3] * x[:, 0] ** 2 \
                   + w[4] * x[:, 0] * x[:, 1] + w[5] * x[:, 1] ** 2
        x1_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
        x2_vals = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
        x_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
        f_grid = linear_quadratic_func(x_grid, w)
        f_grid = f_grid.reshape(x1_grid.shape)
        plt.contour(x1_grid, x2_grid, f_grid, levels=[0.5])
    else:
        x1 = np.linspace(-4, 4, 100)
        x2 = -(w[0] * x1) / w[1]
        plt.plot(x1, x2, 'b', label='Decision boundary')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.show()


if __name__ == '__main__':
    # Parameters of the class-conditional Gaussian pdfs
    m01 = [-1, -1]
    m02 = [1, 1]
    m11 = [-1, 1]
    m12 = [1, -1]
    C = [[1, 0], [0, 1]]

    # Class priors
    p0 = 0.6
    p1 = 0.4

    # Part 1
    samples_, labels_ = gen_samples(10000, False, 'Validate')
    tp_rate, fp_rate, min_tp_rate, min_fp_rate = classify(10000, samples_, labels_)
    plt.scatter(min_fp_rate, min_tp_rate, color='red')
    plt.plot(fp_rate, tp_rate)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Part 2
    use_quadratic = True
    train_samples, train_labels = gen_samples(2000, generate_new=True)
    val_samples, val_labels_ = gen_samples(10000, generate_new=True, kind='Validate')
    weights = train_logistic_linear_function(train_samples, train_labels, quadratic_function=use_quadratic)
    predict(train_samples, weights, train_labels, quadratic_function=use_quadratic)
    predict(val_samples, weights, val_labels_, quadratic_function=use_quadratic)

#     Plot weights as decision boundary, compare linear and quadratic in terms of train error and validation error,
#     check for overfitting, compare theretical optima and this in terms of error

