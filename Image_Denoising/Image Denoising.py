import time

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW

EPSILON = 0.001

plt_directory = ["plt_mvn.png", "plt_mse.png", "plt_ica.png"]
image_index = 0


def get_images(path='train_images.pickle'):
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    return train_pictures


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(patches[:, i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                                                                          window[0] * window[1]).T[:, ::stepsize]


def grayscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = grayscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    global image_index
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
                noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()
    image_index += 1


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    """

    def __init__(self, cov, mix):
        self.cov = cov
        self.mix = mix


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    models - array of GSMs corresponding to the k gaussian's
    """

    def __init__(self, P, vars, mix, models):
        self.P = P
        self.vars = vars
        self.mix = mix
        self.models = models


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return logsumexp(multivariate_normal.logpdf(X.T, mean=model.mean, cov=model.cov))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    sum = 0
    for i in range(len(model.mix)):
        sum += logsumexp(np.log(model.mix[i]) + multivariate_normal.logpdf(X.T, cov=model.cov[i]))
    return sum


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    sum = 0
    S = model.P.T.dot(X)
    for i in range(len(model.mix)):
        sum += GSM_log_likelihood(S[i].reshape(1, -1), model.models[i])
    return sum
    # TODO: YOUR CODE HERE


def wiener_combo(Y, cov, mix, noise_std):
    """
    calculates the waighted some of wiener functions
    """
    x = np.zeros(Y.shape)
    ciy = np.zeros((Y.shape[1], len(mix)))
    if cov.ndim == 1:
        cov = cov[..., np.newaxis]
    for i in range(len(mix)):
        c_v = cov[i] + (np.eye(cov[i].shape[0]) * (noise_std ** 2))
        ciy[:, i] = np.log(mix[i]) + multivariate_normal.logpdf(Y.T, cov=c_v)
    ciy = np.exp(ciy - logsumexp(ciy, axis=1).reshape(-1, 1))
    for i in range(len(mix)):
        x += ciy[:, i].T * wiener(Y, np.zeros(cov.shape[1]), cov[i], noise_std)
    return x


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    mean = X.mean(axis=1)
    cov = np.cov(X)
    return MVN_Model(mean, cov)


def expected_ciy(X, pi_y, cov_y, k):
    ci_y = np.zeros((X.shape[0], k))
    for i in range(k):
        ci_y[:, i] = np.log(pi_y[i]) + multivariate_normal. \
            logpdf(X, np.zeros(X.shape[1]), cov_y[i])
    expectation = np.exp(ci_y - logsumexp(ci_y, axis=1).reshape(-1, 1))
    return expectation


def update_pi_y(ci_y):
    return np.sum(ci_y, axis=0) / ci_y.shape[0]


def update_r(X, ci_y, cov, k):
    r_sq = np.zeros(k)
    ci_y_sum = ci_y.sum(axis=0)
    for i in range(k):
        if cov.ndim == 0:
            a = X.T.dot(((1 / cov) * X))
            r_sq[i] = np.diag(ci_y[:, i] * a).sum() / (X.shape[0] * ci_y_sum[i])
        else:
            r_sq[i] = np.sum(np.diag(ci_y[:, i] * X.T.dot((np.linalg.pinv(cov)).
                                                          dot(X)))) / (X.shape[0] * ci_y_sum[i])
    return r_sq


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    pi_y = np.full(k, 1.0 / k)
    last_r_square = np.zeros((k, 1, 1))
    cov = np.cov(X)
    r_square = np.random.random((k, 1, 1))
    cov_mat = cov * r_square
    last_pi_y = np.zeros(k)
    # using EPSILON to cantroll convergence test
    while not np.all(np.abs(last_r_square - r_square) < EPSILON) or not np.all(np.abs(last_pi_y - pi_y) < EPSILON):
        last_r_square = r_square  # for convergence check
        last_pi_y = pi_y  # for convergence check

        ci_y = expected_ciy(X.T, pi_y, cov_mat, k)

        # find mixture weights
        pi_y = update_pi_y(ci_y)

        # find covariance matrixes
        r_square = update_r(X, ci_y, cov, k)
        cov_mat = np.asarray([r * cov for r in r_square])
    return GSM_Model(cov_mat, pi_y)


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """
    cov = np.cov(X)
    P = np.linalg.eigh(cov)[1]
    S = np.dot(P.T, X)
    models = []
    vars = np.zeros((S.shape[0] * k)).reshape(S.shape[0], k)
    mix = np.zeros((S.shape[0] * k)).reshape(S.shape[0], k)
    for i in range(S.shape[0]):
        models.append(learn_GSM(S[i].reshape(1, -1), k))
        vars[i] = models[i].cov.reshape(-1, k)
        mix[i] = models[i].mix
    return ICA_Model(P, vars, mix, models)



def wiener(y, u, cov, noise_std):
    """
    weiner calculation as should in the EX explanation
    """
    if cov.ndim == 1:
        cov_inv = 1 / cov
    else:
        cov_inv = np.linalg.pinv(cov)
    squr_M = np.eye(cov.shape[0]) * (1 / (noise_std ** 2))
    return (np.linalg.pinv(cov_inv + squr_M)). \
        dot((cov_inv.dot(u)).reshape(len(y), 1) + (1 / (noise_std ** 2)) * y)


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    return wiener(Y, np.zeros(64), mvn_model.cov, noise_std)


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    return wiener_combo(Y, gsm_model.cov, gsm_model.mix, noise_std)

def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    S_clean = np.zeros(Y.shape[0] * Y.shape[1]).reshape(Y.shape)
    Y_in_S_spase = ica_model.P.T.dot(Y)
    for i in range(Y_in_S_spase.shape[0]):
        S_clean[i] = wiener_combo(Y_in_S_spase[i].reshape(1, -1), ica_model.vars[i], ica_model.mix[i], noise_std)
    return ica_model.P.dot(S_clean)


if __name__ == '__main__':
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    train_patches = sample_patches(train_pictures)

    # Unpickle the test images.
    with open('test_images.pickle', 'rb') as f:
        test_picture = pickle.load(f)

    image_running_test_on = 0
    test_patches = sample_patches(test_picture, n=1000)

    # The MVN model.
    mvn_trained_model = learn_MVN(train_patches)
    test_denoising(grayscale_and_standardize(test_picture)[image_running_test_on], mvn_trained_model, MVN_Denoise)

    # The GSM model
    gsm_trained_model = learn_GSM(train_patches, 2)
    test_denoising(grayscale_and_standardize(test_picture)[image_running_test_on], gsm_trained_model, GSM_Denoise)

    # The ICA model.
    ica_trained_model = learn_ICA(train_patches, 2)
    test_denoising(grayscale_and_standardize(test_picture)[image_running_test_on], ica_trained_model, ICA_Denoise)
