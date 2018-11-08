import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
from sklearn.cluster import KMeans
import time


EPSILON = 0.001


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
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
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


def greyscale_and_standardize(images, remove_mean=True):
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
    standardized = greyscale_and_standardize(images, remove_mean)

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
        test_denoising_start_time = time.time()
        denoised_images.append(denoise_image(noisy_images[:, :, i], model, denoise_function,
                                             noise_range[i], patch_size))
        test_denoising_end_time = time.time()
        minutes, seconds = (test_denoising_end_time - test_denoising_start_time) // 60, \
                           (test_denoising_end_time - test_denoising_start_time) % 60
        print("Denoising a image took " + str(minutes) + " minutes and " + str(seconds) + " seconds.")


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


class GMM_Model:
    """
    A class that represents a Gaussian Mixture Model, with all the parameters
    needed to specify the model.

    mixture - a length k vector with the multinomial parameters for the gaussians.
    means - a k-by-D matrix with the k different mean vectors of the gaussians.
    cov - a k-by-D-by-D tensor with the k different covariance matrices.
    """
    def __init__(self, mixture, means, cov):
        self.mixture = mixture
        self.means = means
        self.cov = cov


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    gmm - a GMM_Model object.
    """
    def __init__(self, mean, cov, gmm):
        self.mean = mean
        self.cov = cov
        self.gmm = gmm


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    gmm - a GMM_Model object.
    """
    def __init__(self, cov, mix, gmm):
        self.cov = cov
        self.mix = mix
        self.gmm = gmm


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    gmms - A list of K GMM_Models, one for each source.
    """
    def __init__(self, P, vars, mix, gmms):
        self.P = P
        self.vars = vars
        self.mix = mix
        self.gmms = gmms


def weiner(noise_patch, mean, covs, var):
    """
    This function calculates the weiner filter.
    :param noise_patch: The noise patch.
    :param mean: The mean.
    :param covs: The covariance matrix.
    :param var: The variance of the noise.
    :return: The estimation of the clean patch.
    """
    inv_covs = np.linalg.pinv(covs)
    inv_var_squared = 1/(var ** 2)
    normal_sigma_squared_I = np.eye(covs.shape[0])
    np.fill_diagonal(normal_sigma_squared_I, inv_var_squared)
    return (np.linalg.pinv(inv_covs+normal_sigma_squared_I)).\
        dot((inv_covs.dot(mean)).reshape(len(noise_patch), 1) + inv_var_squared * noise_patch)


def weiner_combination(noise_patch, model, var):
    """
    This function is the helper function to calculate a combination of weiner filters.
    :param noise_patch: The noise patch
    :param model: The trained model.
    :param var: The variance of the noise.
    :return: The estimated clean image.
    """
    estimated_x = np.zeros(noise_patch.shape)
    c = np.zeros((noise_patch.shape[1], len(model.mixture)))
    for i in range(len(model.mixture)):
        cov_plus_var = model.cov[i] + (np.eye(model.cov[i].shape[0]) * (var ** 2))
        c[:, i] = np.log(model.mixture[i]) + \
                 scipy.stats.multivariate_normal.logpdf(noise_patch.T, model.means[i], cov_plus_var)
    c = np.exp(c - logsumexp(c, axis=1).reshape(-1, 1))
    for i in range(len(model.mixture)):
        estimated_x += c[:, i].T * weiner(noise_patch, model.means[i], model.cov[i], var)
    return estimated_x


def log_likelihood(X, model):
    """
    This function is a helper function for calculating the log likelihood.
    :param X: The data.
    :param model: The model we ar training.
    :return: The sum of the log likelihood's.
    """
    k = model.mixture.shape[0]
    c = np.zeros((X.shape[1], k))
    for y in range(k):
        c[:, y] = np.log(model.mixture[y]) + scipy.stats.multivariate_normal.\
            logpdf(X.T, model.means[y], model.cov[y], allow_singular=True)
    return logsumexp(c, axis=1).sum()


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return log_likelihood(X, model.gmm)


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    return log_likelihood(X, model.gmm)


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """
    log_likelihood_sum = 0
    X = model.P.T.dot(X)
    for i in range(len(model.mix)):
        log_likelihood_sum += log_likelihood(X[i].reshape(1, -1), model.gmms[i])
    return log_likelihood_sum


def E_step(c, X, k, model):
    """
    This function is the E-step of the EM algorithm.
    :param c: This a empty numpy array of the shape(X.shape[1], k).
    :param X: The data.
    :param k: The number of gaussians we are learning.
    :param model: The model we are training.
    :return: The c that is calculated during the E-step (where c[i,y] is the likelihood that the i data
                sample is from the y'th gaussian.
    """
    for y in range(k):
        c[:, y] = np.log(model.mixture[y]) + scipy.stats.multivariate_normal.\
            logpdf(X.T, model.means[y], model.cov[y], allow_singular=True)
    return np.exp(c - logsumexp(c, axis=1).reshape(-1, 1))


def M_step(c, X, k, model,learn_mixture=True, learn_means=True, learn_covariances=True):
    """
    This function is the M-step of the EM algorithm.
    :param c: The c that is calculated during the E-step (where c[i,y] is the likelihood that the i data
                sample is from the y'th gaussian.
    :param X: The data.
    :param k: The number of the gaussians we are learning.
    :param model: The model we are training.
    :param learn_mixture: A boolean representing if to learn the mixture.
    :param learn_means: A boolean representing if to learn the means.
    :param learn_covariances: A boolean representing if to learn the covariances.
    :return: The model after maximizing using the M-step.
    """
    c_sum_on_i = c.sum(axis=0)
    if learn_mixture:
        model.mixture = (c_sum_on_i / X.shape[1])
    if learn_means:
        for y in range(k):
            model.means[y] = (c[:, y] * X).sum(axis=1) / c_sum_on_i[y]
    if learn_covariances:
        for y in range(k):
            x_minus_mean_y = X - model.means[y].reshape(-1, 1)
            model.cov[y] = (c[:, y] * x_minus_mean_y).dot(x_minus_mean_y.T) / c_sum_on_i[y]
    return model


def learn_GMM(X, k, initial_model, learn_mixture=True, learn_means=True,
              learn_covariances=True, iterations=10):
    """
    A general function for learning a GMM_Model using the EM algorithm.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: number of components in the mixture.
    :param initial_model: an initial GMM_Model object to initialize EM with.
    :param learn_mixture: a boolean for whether or not to learn the mixtures.
    :param learn_means: a boolean for whether or not to learn the means.
    :param learn_covariances: a boolean for whether or not to learn the covariances.
    :param iterations: Number of EM iterations (default is 10).
    :return: (GMM_Model, log_likelihood_history)
            GMM_Model - The learned GMM Model.
            log_likelihood_history - The log-likelihood history for debugging.
    """

    c = np.zeros((X.shape[1], k))
    log_likelihood_history = []
    converged = False
    i = 0
    while not converged and i < iterations:
        old_data = [initial_model.mixture.copy(), initial_model.means.copy(), initial_model.cov.copy()]
        c = E_step(c, X, k, initial_model)
        initial_model = M_step(c, X, k, initial_model, learn_mixture, learn_means, learn_covariances)
        converged = np.all((old_data[0] - initial_model.mixture) < EPSILON) \
                    and np.all((old_data[1] - initial_model.means) < EPSILON) \
                    and np.all((old_data[2] - initial_model.cov) < EPSILON)
        log_likelihood_history.append(log_likelihood(X, initial_model))
        i += 1
    return initial_model, log_likelihood_history


def get_initial_model(X, k):
    """
    This function returns a estimation for a good initialization by running k-means on the data and returning
    The means to by the cluster centers.
    :param X: The data.
    :param k: The number of clusters.
    :return: A GMM model of a estimated good initialization.
    """
    data = KMeans(k).fit(X)
    covs = np.zeros((k, X.shape[1], X.shape[1]))
    if len(data.cluster_centers_[0]) == 1:
        for i in range(k):
            covs[i] = np.eye(X.shape[1]) * np.random.uniform()
    else:
        for i in range(k):
            covs[i] = np.cov(data.cluster_centers_[i].T)
    return GMM_Model(np.array([1/k]*k), data.cluster_centers_, covs)


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    trained_gmm = learn_GMM(X, 1, get_initial_model(X.T, 1), True, True, True)[0]
    return MVN_Model(trained_gmm.means, trained_gmm.cov, trained_gmm)


def learn_GSM(X, k, iterations=10):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :param iterations: This is the maximum number of iterations to do until GSM converges.
    :return: A trained GSM_Model object.
    """
    c = np.zeros((X.shape[1], k))
    initial_model = get_initial_model(X.T, k)
    cov = np.cov(X)
    initial_model = GSM_Model(cov, np.random.uniform(size=k), initial_model)
    for i in range(k):
        initial_model.gmm.means[i] = np.array([0] * cov.shape[0])
        initial_model.gmm.cov[i] = initial_model.cov * initial_model.mix[i]
    converged = False
    i = 0
    while not converged and i < iterations:
        old_data = [initial_model.gmm.mixture.copy(), initial_model.mix.copy()]
        c = E_step(c, X, k, initial_model.gmm)
        initial_model.gmm = M_step(c, X, k, initial_model.gmm, True, False, False)
        for y in range(k):
            initial_model.mix[y] = np.diag(c[:, y] * X.T.dot((np.linalg.pinv(initial_model.cov)).
                                                             dot(X))).sum() / (X.shape[0] * c.sum(axis=0)[y])
            initial_model.gmm.cov[y] = initial_model.cov * initial_model.mix[y]
        converged = np.all((old_data[0] - initial_model.gmm.mixture) < EPSILON) \
                    and np.all((old_data[1] - initial_model.mix) < EPSILON)
        i += 1
    return initial_model


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
    cov_matrix = np.cov(X)
    _, P = np.linalg.eigh(cov_matrix)
    ica_model = ICA_Model(P, np.zeros((X.shape[0], k)), np.zeros((X.shape[0], k)), [])
    X = P.T.dot(X)
    for i in range(X.shape[0]):
        ica_model.gmms.append(learn_GMM(X[i].reshape(1, -1), k,
                                        get_initial_model(X[i].reshape(-1, 1), k), True, True, True)[0])
        ica_model.vars[i] = ica_model.gmms[i].cov.reshape(k, )
        ica_model.mix[i] = ica_model.gmms[i].mixture
    return ica_model


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
    mvn_model.gmm.means = mvn_model.mean
    return weiner_combination(Y, mvn_model.gmm, noise_std)


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
    return weiner_combination(Y, gsm_model.gmm, noise_std)


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

    s_noisy = ica_model.P.T.dot(Y)
    for i in range(s_noisy.shape[0]):
        Y[i] = weiner_combination(s_noisy[i].reshape(1, -1), ica_model.gmms[i], noise_std)
    return ica_model.P.dot(Y)


def save_model(model, title):
    """
    This function pickle's a model.
    :param model: The model to pickle.
    :param title: The title of the pickle.
    :return: Nothing
    """
    with open(title + ".pickle", "wb") as file:
        pickle.dump(model, file)


def load_model(title):
    """
    This function pickle's a model.
    :param title: The title of the pickle.
    :return: The unpickled model.
    """
    with open(title + ".pickle", "rb") as file:
        model = pickle.load(file)
    return model


# if __name__ == '__main__':
#     # Unpickle the training images and creating a data set.
#     with open('train_images.pickle', 'rb') as f:
#         train_pictures = pickle.load(f)
#     train_patches = sample_patches(train_pictures)
#
#     # Unpickle the test images.
#     with open('test_images.pickle', 'rb') as f:
#         test_picture = pickle.load(f)
#     image_running_test_on = 0
#     test_patches = sample_patches(test_picture)
#
#     # Test EM and get log likelihood graph.
#     _, log_likelihood_history = learn_GMM(train_patches, 15, get_initial_model(train_patches.T, 15))
#     plt.plot(log_likelihood_history)
#     plt.show()
#
#     # The MVN model.
#     training_start_time = time.time()
#     mvn_trained_model = learn_MVN(train_patches)
#     training_end_time = time.time()
#     minutes, seconds = (training_end_time - training_start_time) // 60, \
#                        (training_end_time - training_start_time) % 60
#     print("Training the MVN model took " + str(minutes) + " minutes and " + str(seconds) + " seconds.")
#     test_denoising(greyscale_and_standardize(test_picture)[image_running_test_on], mvn_trained_model, MVN_Denoise)
#     print("MVN's log likelihood is: " + str(MVN_log_likelihood(test_patches, mvn_trained_model)))
#
#     # The GSM model.
#     training_start_time = time.time()
#     gsm_trained_model = learn_GSM(train_patches, 5)
#     training_end_time = time.time()
#     minutes, seconds = (training_end_time - training_start_time) // 60, \
#                        (training_end_time - training_start_time) % 60
#     print("Training the GSM model took " + str(minutes) + " minutes and " + str(seconds) + " seconds.")
#     test_denoising(greyscale_and_standardize(test_picture)[image_running_test_on], gsm_trained_model, GSM_Denoise)
#     print("GSM's log likelihood is: " + str(GSM_log_likelihood(test_patches, gsm_trained_model)))
#
#     # The ICA model.
#     training_start_time = time.time()
#     ica_trained_model = learn_ICA(train_patches, 5)
#     training_end_time = time.time()
#     minutes, seconds = (training_end_time - training_start_time) // 60, \
#                        (training_end_time - training_start_time) % 60
#     print("Training the ICA model took " + str(minutes) + " minutes and " + str(seconds) + " seconds.")
#     test_denoising(greyscale_and_standardize(test_picture)[image_running_test_on], ica_trained_model, ICA_Denoise)
#     print("ICA's log likelihood is: " + str(ICA_log_likelihood(test_patches, ica_trained_model)))

def get_images(path='train_images.pickle'):
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)
    return train_pictures

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



if __name__ == '__main__':
    images = get_images()
    X = sample_patches(images)
    model = learn_ICA(sample_patches(images),2)
    test_denoising(grayscale_and_standardize(images)[0], model, ICA_Denoise)