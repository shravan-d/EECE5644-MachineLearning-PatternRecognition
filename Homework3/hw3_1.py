import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn


# Generate samples
def gen_samples(count, generate_new=False, kind='Train'):
    if generate_new:
        print('Generating samples of length: ', count)
        generated_samples = np.zeros((count, 3))
        true_labels = np.zeros(count)
        for i in range(count):
            rand = np.random.uniform()
            if rand < 0.25:
                generated_samples[i, :] = np.random.multivariate_normal(m0, C)
                true_labels[i] = 0
            elif 0.25 <= rand < 0.5:
                generated_samples[i, :] = np.random.multivariate_normal(m1, C)
                true_labels[i] = 1
            elif 0.5 <= rand < 0.75:
                generated_samples[i, :] = np.random.multivariate_normal(m2, C)
                true_labels[i] = 2
            else:
                generated_samples[i, :] = np.random.multivariate_normal(m3, C)
                true_labels[i] = 3
        np.save('data_hw3/'+kind + 'Samples' + str(count), generated_samples)
        np.save('data_hw3/'+kind + 'Labels' + str(count), true_labels)
        return np.array(generated_samples), np.array(true_labels)
    else:
        samples = np.load('data_hw3/'+kind + 'Samples' + str(count) + '.npy')
        labels = np.load('data_hw3/'+kind + 'Labels' + str(count) + '.npy')
        return samples, labels


def plot3(a, b, c, labels=None, kind='Training', col="b"):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    markers = ['o', 'x', '^', 'X']
    colors = ['b', 'g', 'r', 'y']
    for i in range(labels.shape[0]):
        ax.scatter(a[i], b[i], c[i], marker=markers[int(labels[i])-1], color=colors[int(labels[i])-1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(kind + ' Dataset')
    plt.show()


class MLP(nn.Module):
    def __init__(self, n_input, P, C):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, P),
            # nn.ReLU(),
            nn.Softplus(),
            nn.Linear(P, C)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def theoretical_classifier(samples, labels):
    N = len(labels)
    predictions = []
    for sample in tqdm(samples):
        posteriors = [multivariate_normal.pdf(sample, mean=means[class_], cov=C) for class_ in range(4)]
        predictions.append(np.argmax(posteriors))
    error = sum([predictions[i] != labels[i] for i in range(N)]) / N
    print('Theoretical Probability of Error: ', error)


def train(P, samples, labels):
    N = len(labels)
    model = MLP(3, P, 4)
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


def k_fold(samples_, labels_, k=5, total_perceptrons=100):
    errors_total, perceptrons_total = [], []
    fig, ax = plt.subplots(figsize=(10, 8))
    for samples, labels in zip(samples_, labels_):
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
        errors_total.append(errors_)
        perceptrons_total.append(perceptrons)
        print('{} perceptrons result in minimum error of {:.4f}'.format(min_error_neurons, min_error))

    for i in range(len(samples_)):
        ax.plot(perceptrons_total[i], errors_total[i], label="N = {}".format(len(samples_[i])))
    ax.set_title("Cross-Validation -> Mean probability of error vs Number of perceptrons")
    ax.set_xlabel('Perceptrons')
    ax.set_ylabel("Pr(error)")
    ax.legend()
    plt.show()


def training(samples, labels):
    model_info = {'best_model': None, 'loss': np.Inf}
    for i in range(10):
        model, loss = train(2, torch.FloatTensor(samples), torch.LongTensor(labels))
        if loss < model_info['loss']:
            model_info = {'best_model': model, 'loss': loss}
    print('Best Model has loss of ', model_info['loss'])
    return model_info['best_model']


def testing(samples, labels, model):
    pred = predict(model, torch.FloatTensor(samples))
    error = (sum([pred[i] != labels[i] for i in range(labels.shape[0])]) /
             labels.shape[0])
    print('Test set has loss of ', error)
    return error


if __name__ == '__main__':
    # Parameters of the class-conditional Gaussian pdfs
    priors = [0.25, 0.25, 0.25, 0.25]
    m0 = [-1, -1, 1]
    m1 = [1, 1, 2]
    m2 = [-1, 1, 3]
    m3 = [1, -1, 4]
    means = [m0, m1, m2, m3]
    C = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    training_counts = [100, 200, 500, 1000, 2000, 5000]

    training_count = 100
    train_data, train_labels = gen_samples(training_count, generate_new=False)
    plot3(train_data[:, 0], train_data[:, 1], train_data[:, 2], train_labels)
    theoretical_classifier(train_data, train_labels)
    # k_fold([train_data, train_data1, train_data2, train_data3, train_data4, train_data5],
    #        [train_labels, train_labels1, train_labels2, train_labels3, train_labels4, train_labels5],
    #        total_perceptrons=30)

    test_data, test_labels = gen_samples(100000, generate_new=False)
    sizes = [100, 200, 500, 1000, 2000, 5000]
    errors = []
    for training_count in sizes:
        train_data, train_labels = gen_samples(training_count, generate_new=False)
        model = training(train_data, train_labels)
        error = testing(test_data, test_labels, model)
        errors.append(error)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sizes, errors)
    ax.set_title("Mean probability of error on test set vs Training set size")
    ax.set_xlabel('Training set size')
    ax.set_ylabel("Pr(error)")
    ax.legend()
    plt.show()
