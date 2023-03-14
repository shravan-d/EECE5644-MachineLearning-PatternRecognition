import numpy as np
import matplotlib.pyplot as plt


def get_true_coords():
    r = np.sqrt(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2) * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])


def get_range_measurements(count, xy_true):
    new_coords = [generate_landmarks(i, count) for i in range(K)]
    distances = [generate_measurement(new_coords[i], xy_true) for i in range(K)]
    return new_coords, distances


def generate_landmarks(i, count):
    angle = 2 * np.pi / count * i
    return np.array([np.cos(angle), np.sin(angle)])


def generate_measurement(xy_landmark, xy_true):
    dTi = np.linalg.norm(xy_true-xy_landmark)
    while True:
        noise = np.random.normal(0, 0.3)
        measurement = dTi + noise
        if measurement >= 0:
            return measurement


def plot_equilevels(landmarks, distances, xy_true):
    gridpoints = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    contour_values = MAP_estimate(gridpoints, landmarks, distances)
    ax = plt.gca()

    unit_circle = plt.Circle((0, 0), 1, color='red', fill=False)
    ax.add_artist(unit_circle)

    plt.contour(gridpoints[0], gridpoints[1], contour_values, cmap='plasma_r', levels=np.geomspace(0.00001, 20, 100))

    for i in range(K):
        xy_i = landmarks[i]
        plt.plot(xy_i[0], xy_i[1], 'o', color='green')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MAP estimation contours for K = " + str(K))
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.plot([xy_true[0]], [xy_true[1]], 'x', color='r')
    plt.show()


def MAP_estimate(xy, landmarks, distance):
    xy = np.expand_dims(np.transpose(xy, axes=(1, 2, 0)), axis=len(np.shape(xy))-1)
    covariance = np.array([[0.25**2, 0], [0, 0.25**2]])

    prior = np.matmul(xy, np.linalg.inv(covariance))
    prior = np.matmul(prior, np.swapaxes(xy, 2, 3))
    prior = np.squeeze(prior)
    prior_ = xy.dot(np.linalg.inv(covariance)).dot(np.swapaxes(xy, 2, 3))

    range_sum = 0
    for i in range(K):
        xy_i = landmarks[i]
        r_i = distance[i]
        d_i = np.linalg.norm(xy - xy_i[None, None, None, :], axis=3)
        range_sum += np.squeeze((r_i - d_i)**2 / 0.3**2)

    return prior + range_sum


if __name__ == '__main__':
    xy_true = get_true_coords()
    for K in [1, 2, 3, 4]:
        landmark, dist = get_range_measurements(K, xy_true)
        plot_equilevels(landmark, dist, xy_true)
