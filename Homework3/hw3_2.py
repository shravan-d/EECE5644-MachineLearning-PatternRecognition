import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from sklearn import mixture
import collections
from tqdm import tqdm


# Generate samples
def gen_samples(count, generate_new=False, kind='Train'):
    if generate_new:
        print('Generating samples of length: ', count)
        generated_samples = np.zeros((count, 2))
        true_labels = np.zeros(count)
        for i in range(count):
            rand = np.random.uniform()
            if rand < priors[0]:
                generated_samples[i, :] = np.random.multivariate_normal(m0, C0)
                true_labels[i] = 0
            elif priors[0] <= rand < priors[0] + priors[1]:
                generated_samples[i, :] = np.random.multivariate_normal(m1, C1)
                true_labels[i] = 1
            elif priors[0] + priors[1] <= rand < priors[0] + priors[1] + priors[2]:
                generated_samples[i, :] = np.random.multivariate_normal(m2, C2)
                true_labels[i] = 2
            else:
                generated_samples[i, :] = np.random.multivariate_normal(m3, C3)
                true_labels[i] = 3
        np.save('data_hw3/'+kind + '2Samples' + str(count), generated_samples)
        np.save('data_hw3/'+kind + '2Labels' + str(count), true_labels)
        return np.array(generated_samples), np.array(true_labels)
    else:
        samples = np.load('data_hw3/'+kind + '2Samples' + str(count) + '.npy')
        labels = np.load('data_hw3/'+kind + '2Labels' + str(count) + '.npy')
        return samples, labels


def plot(a, b, labels=None, kind='Training'):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    markers = ['o', 'x', '^', 'X']
    colors = ['b', 'g', 'r', 'y']
    for i in range(labels.shape[0]):
        ax.scatter(a[i], b[i], marker=markers[int(labels[i])-1], color=colors[int(labels[i])-1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(kind + ' Dataset')
    plt.show()


def split_into_k(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def fit_models(samples, labels, k=10):
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0) for n in range(1, 7)]
    scores_total = []
    N = len(labels)
    samples_k = list(split_into_k(np.arange(N), k))
    for model in models:
        scores = []
        for k_idx in range(k):
            validation_samples = np.array([samples[i] for i in range(N) if i in samples_k[k_idx]])
            validation_labels = np.array([labels[i] for i in range(N) if i in samples_k[k_idx]])
            training_samples = np.array([samples[i] for i in range(N) if i not in samples_k[k_idx]])
            training_labels = np.array([labels[i] for i in range(N) if i not in samples_k[k_idx]])
            model.fit(training_samples)
            score = model.score(validation_samples)
            scores.append(score)
        mean_score = np.mean(scores)
        scores_total.append(-1*mean_score)
        # print('Model has mean score of: ', mean_score)

    plt.plot(np.arange(1, 7), scores_total, label='negative log_likelihood')
    plt.xlabel('Number of components')
    plt.ylabel('Negative Log Likelihood score')
    plt.legend(loc='best')
    plt.title('Negative log_likelihood score for different GMM models')
    plt.show()
    return np.argmin(scores_total) + 1


if __name__ == '__main__':
    priors = [0.2, 0.25, 0.35, 0.3]
    m0 = [-1, -1]
    m1 = [-0.5, 1]
    m2 = [1, 0.5]
    m3 = [1, -1]
    C0 = [[0.25, 0.15], [0.15, 0.25]]
    C1 = [[0.1, 0.04], [0.04, 0.1]]
    C2 = [[0.2, -0.15], [-0.15, 0.2]]
    C3 = [[0.35, 0.1], [0.1, 0.35]]

    sizes = [10, 100, 1000, 10000]
    reiterations = 30
    size_components = {size: [] for size in sizes}
    for _ in tqdm(range(reiterations)):
        for size in sizes:
            data, labels_ = gen_samples(size, generate_new=True)
            plot(data[:, 0], data[:, 1], labels_)
            best_model_components = fit_models(data, labels_)
            size_components[size].append(best_model_components)
    for size_component in size_components.keys():
        counter = collections.Counter(size_components[size_component])
        print('The dataset size of {} has the following frequencies of number components {}'
              .format(size_component, counter))
