import numpy as np
from matplotlib import pyplot as plt
from sklearn import mixture
from tqdm import tqdm
import cv2


def show_image_grid(images, n_components, title=''):
    _, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax, n in zip(images, axs, n_components):
        if len(img.shape) == 3:
            ax.imshow(img, vmin=0, vmax=255)
            if n == -2:
                ax.set_title("Original Image")
            else:
                ax.set_title(str(n+2)+' Components')
        else:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    # plt.title(title)
    plt.show()


def color_segments(predictions):
    segmented_image = predictions.reshape(image.shape[0], image.shape[1])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128)]
    image_temp = np.zeros(image.shape)
    for i in range(segmented_image.shape[0]):
        for j in range(segmented_image.shape[1]):
            image_temp[i, j] = colors[segmented_image[i, j]]
    return image_temp


def get_features():
    features = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            current_row = [i, j, image[i, j][0], image[i, j][1], image[i, j][2]]
            features.append(current_row)
    features = np.array(features)
    features_norm = (features - np.min(features, axis=0)) / features.ptp(0)
    return features_norm


def split_into_k(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def fit_models(data, k=10):
    models = [mixture.GaussianMixture(n_components=n, max_iter=400, tol=1e-3) for n in range(2, 9)]
    scores_total = []
    N = len(data)
    samples_k = list(split_into_k(np.arange(N), k))
    for model in tqdm(models):
        scores = []
        for k_idx in range(k):
            validation_pixels = np.array([data[i] for i in range(N) if i in samples_k[k_idx]])
            training_pixels = np.array([data[i] for i in range(N) if i not in samples_k[k_idx]])
            model.fit(training_pixels)
            score = model.score(validation_pixels)
            scores.append(score)
        mean_score = np.mean(scores)
        scores_total.append(-1*mean_score)
        # print('Model has mean score of: ', mean_score)

    plt.plot(np.arange(2, 9), scores_total, label='negative log_likelihood')
    plt.xlabel('Number of components')
    plt.ylabel('Negative Log Likelihood score')
    plt.legend(loc='best')
    plt.title('Negative log_likelihood score for different GMM models')
    # plt.show()
    return np.argpartition(scores_total, 3)


def segment_best_model(n_components):
    images = [image]
    for n in n_components:
        gmm = mixture.GaussianMixture(n_components=n+2, max_iter=400, tol=1e-3)
        gmm_predictions = gmm.fit_predict(feature_vector)
        segmented = color_segments(gmm_predictions)
        images.append(segmented)
    n_components = np.insert(n_components, 0, -2, axis=0)
    show_image_grid(images, n_components)


if __name__ == '__main__':
    image = cv2.imread('388016.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    width = int(image.shape[1] * 0.5)
    height = int(image.shape[0] * 0.5)
    # image = cv2.resize(image, (width, height))
    feature_vector = get_features()
    best_n_components = fit_models(feature_vector)
    segment_best_model(best_n_components)
