import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.svm import SVC


def generate_data(N):
    samples = np.zeros((N, 2))
    labels = np.ones(N)
    for i in range(N):
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        uniform_component = np.array([np.cos(theta), np.sin(theta)]).T
        if np.random.rand() < prior:
            labels[i] = 0
            samples[i] = 2 * uniform_component + multivariate_normal.rvs(mean=0, cov=1)
        else:
            samples[i] = 4 * uniform_component + multivariate_normal.rvs(mean=0, cov=1)
    return samples, labels


def plot(a, b, labels=None, pred_labels=None, kind='Training'):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    markers = ['o', 'x', '^', 'X']
    colors = ['b', 'g', 'r', 'y']
    for i in range(labels.shape[0]):
        if pred_labels is None:
            ax.scatter(a[i], b[i], marker=markers[int(labels[i])-1], color=colors[int(labels[i])-1])
        else:
            if pred_labels[i] == labels[i]:
                ax.scatter(a[i], b[i], marker=markers[int(labels[i]) - 1], color='g')
            else:

                ax.scatter(a[i], b[i], marker=markers[int(labels[i]) - 1], color='r')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(kind + ' Dataset')
    plt.show()


class MLP(nn.Module):
    def __init__(self, n_input, P, C):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, P),
            nn.ReLU(),
            nn.Linear(P, C)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def train(P, samples, labels):
    N = len(labels)
    model = MLP(2, P, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    n_epochs, losses = 10, []
    model.train()
    for epoch in range(n_epochs):
        running_loss = 0
        for sample, label in zip(samples, labels):
            optimizer.zero_grad()
            pred = model(sample)
            loss = loss_function(pred, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        # print('Loss: ', running_loss / N)
        losses.append(running_loss / N)
    # print('Training Done')
    return model, np.mean(losses)


def predict(model, samples):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sample in samples:
            pred = model(sample)
            predictions.append(np.argmax(pred.numpy()))
    return predictions


def split_into_k(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def k_fold(samples, labels, k=10, total_perceptrons=10):
    fig, ax = plt.subplots(figsize=(10, 8))
    N = len(labels)
    samples_k = list(split_into_k(np.arange(N), k))
    min_error, min_error_neurons = np.Inf, -1
    errors_, perceptrons = [], []
    for p in tqdm(range(1, total_perceptrons+1)):
        errors = []
        for k_idx in range(k):
            validation_samples = np.array([samples[i] for i in range(N) if i in samples_k[k_idx]])
            validation_labels = np.array([labels[i] for i in range(N) if i in samples_k[k_idx]])
            training_samples = np.array([samples[i] for i in range(N) if i not in samples_k[k_idx]])
            training_labels = np.array([labels[i] for i in range(N) if i not in samples_k[k_idx]])
            model, _ = train(p, torch.FloatTensor(training_samples), torch.LongTensor(training_labels))
            pred = predict(model, torch.FloatTensor(validation_samples))
            error = (sum([pred[i] != validation_labels[i] for i in range(validation_labels.shape[0])]) /
                     validation_labels.shape[0])
            errors.append(error)
        mean_error = np.mean(errors)
        perceptrons.append(p)
        errors_.append(mean_error)
        if mean_error < min_error:
            min_error = mean_error
            min_error_neurons = p
    print('{} perceptrons result in minimum error of {:.4f}'.format(min_error_neurons, min_error))

    ax.plot(perceptrons, errors_)
    ax.set_title("Cross-Validation -> Mean probability of error vs Number of perceptrons")
    ax.set_xlabel('Perceptrons')
    ax.set_ylabel("Pr(error)")
    ax.legend()
    plt.show()

    return np.argmin(errors_) + 1


def training(samples, labels, best_n_perceptrons=None):
    model_info = {'best_model': None, 'loss': np.Inf}
    for i in range(10):
        model, loss = train(best_n_perceptrons, torch.FloatTensor(samples), torch.LongTensor(labels))
        if loss < model_info['loss']:
            model_info = {'best_model': model, 'loss': loss}
    print('Best Model has loss of ', model_info['loss'])
    return model_info['best_model']


def testing(samples, labels, model):
    pred = predict(model, torch.FloatTensor(samples))
    error = (sum([pred[i] != labels[i] for i in range(labels.shape[0])]) /
             labels.shape[0])
    print('Test set has loss of ', error)
    return error, pred


def SVM_train(overlap, kernel, samples, labels):
    model = SVC(C=overlap, gamma=1/(2*kernel**2))
    model.fit(samples, labels)
    return model


def SVM_test(model, test_samples, test_labels):
    predictions = model.predict(test_samples)
    error = (sum([predictions[i] != test_labels[i] for i in range(test_labels.shape[0])]) /
             test_labels.shape[0])
    print('Test set has loss of ', error)
    return predictions, error
    

def k_fold_SVM(samples, labels, k=10):
    fig, ax = plt.subplots(figsize=(10, 8))
    overlaps = np.geomspace(0.001, 100, 6)
    kernels = np.geomspace(0.001, 100, 6)

    min_error, min_error_overlap, min_error_kernel = np.Inf, -1, -1
    error_map = np.zeros((overlaps.shape[0], kernels.shape[0]))

    for i, overlap in enumerate(overlaps):
        for j, kernel in enumerate(kernels):
            errors = []
            N = len(labels)
            samples_k = list(split_into_k(np.arange(N), k))
            for k_idx in range(k):
                validation_samples = np.array([samples[i] for i in range(N) if i in samples_k[k_idx]])
                validation_labels = np.array([labels[i] for i in range(N) if i in samples_k[k_idx]])
                training_samples = np.array([samples[i] for i in range(N) if i not in samples_k[k_idx]])
                training_labels = np.array([labels[i] for i in range(N) if i not in samples_k[k_idx]])
                model = SVM_train(overlap, kernel, training_samples, training_labels)
                pred, error = SVM_test(model, validation_samples, validation_labels)
                errors.append(error)
            mean_error = np.mean(errors)
            error_map[i, j] = mean_error
            if mean_error < min_error:
                min_error = mean_error
                min_error_overlap = overlap
                min_error_kernel = kernel

    print('Overlap value of {} and kernel width of {} result in minimum error of {:.4f}'
          .format(min_error_overlap, min_error_kernel, min_error))

    ax.set_title('Mean error across k-validation for different hyperparameters')
    ax.imshow(error_map, interpolation='None', cmap="GnBu" )

    for i in range(len(overlaps)):
        for j in range(len(kernels)):
            text = 'Overlap: {}\nKernel Width: {}\nError: {:2f}'.format(overlaps[i], kernels[j], error_map[i, j])
            ax.text(j, i, text, va='center', ha='center')

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return min_error_overlap, min_error_kernel


if __name__ == '__main__':
    prior = 0.5

    samples, labels = generate_data(1000)
    plot(samples[:, 0], samples[:, 1], labels)

    # MLP
    best_n_perceptrons = k_fold(samples, labels, total_perceptrons=20)
    MLP_model = training(samples, labels, best_n_perceptrons)

    # SVM
    C, k_w = k_fold_SVM(samples, labels)
    SVM_model = SVM_train(C, k_w, samples, labels)

    test_samples, test_labels = generate_data(10000)
    plot(test_samples[:, 0], test_samples[:, 1], test_labels, kind='Test')

    # MLP
    MLP_error, MLP_pred = testing(test_samples, test_labels, MLP_model)

    # SVM
    SVM_pred, SVM_error = SVM_test(SVM_model, test_samples, test_labels)

    plot(test_samples[:, 0], test_samples[:, 1], test_labels, SVM_pred, 'SVM - Prediction')
    plot(test_samples[:, 0], test_samples[:, 1], test_labels, MLP_pred, 'MLP - Prediction')
