import numpy as np
import cv2 as cv
from typing import List
from abc import ABCMeta, abstractmethod


class FourierDescriptorBase(metaclass=ABCMeta):
    """
    Base class to construct Fourier descriptor of given binary images \n
    @param list of filenames or list of images
    @param normalize set if Fourier descriptors should be normalized
    @param number of fourier descriptor (FD) harmonics -> FD[-m_n, -m_n-1, ...-m_1, m_0, m_1, ...., m_n-1, m_n ] for n harmonics
    """

    def __init__(self, images: List, descriptor_harmonics: int, normalize:bool = True):
        self.descriptors = []
        self.contours = []
        self.descriptor_harmonics = descriptor_harmonics
        self._normalize = normalize
        if isinstance(images[0], str):
            self.images = [cv.imread(image, cv.IMREAD_GRAYSCALE)
                           for image in images]
        else:
            self.images = images

    def run(self) -> np.ndarray:
        for image in self.images:
            self.contours.append(
                self.detectOutlineContour(image.astype('uint8')))
            unnormalized_descriptor = self.makeFourierDescriptorFromPolygon(
                self.contours[-1][:, 0], self.contours[-1][:, 1], self.descriptor_harmonics)
            if self._normalize:
                self.descriptors.append(
                    self.normalizeDescriptor(unnormalized_descriptor))
            else:
                self.descriptors.append(unnormalized_descriptor)

        self.descriptors = np.array(self.descriptors)
        return self.descriptors

    def detectOutlineContour(self, bin_image: np.ndarray) -> np.ndarray:
        """ Detects outline contour of given binary image """
        contours, _ = cv.findContours(
            bin_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return self.getLargestContour(contours)

    def makeFourierDescriptor_basic(self, contour: List):
        """ calculates the (unnormalized!) fourier descriptor from a list of points """
        contour_complex = np.empty(contour.shape[:-1], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]
        return np.fft.fft(contour_complex)

    @abstractmethod
    def normalizeDescriptor(self, descriptor: np.ndarray) -> np.ndarray:
        pass

    def getLargestContour(self, contours: List) -> np.ndarray:
        """ Returns the largest contour of a list of contours. 
            Makes sure that contours created of noisy image artifacts are not returned
        """
        contours = sorted(contours, key=len, reverse=True)
        return np.array(contours[0][:, 0, :])

    def makeFourierDescriptorFromPolygon(self, X: List, Y: List, harmonics: int = 40) -> np.ndarray:
        """ @brief  Compute the Fourier Descriptors for a polygon.
                    Implements Kuhl and Giardina method of computing the coefficients
                    for a specified number of harmonics. See the original paper for more detail:
                    Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of a closed
                    contour. Computer graphics and image processing, 18(3), 236-258.
                    Or see W. Burger et. al. - Principles of Digital Image Processing - Advanced Methods
            @param X (list): A list (or numpy array) of x coordinate values.
            @param Y (list): A list (or numpy array) of y coordinate values.
            @param harmonics (int): The number of harmonics to compute for the given
                    shape, defaults to 10.
            @return numpy.ndarray: A complex numpy array of shape (harmonics, ) representing the unnormalized Fourier descriptor 
        """
        new_vector_length = 2*harmonics+1
        FD = np.zeros(new_vector_length)+1j*np.zeros(new_vector_length)

        contour = np.array([(x, y) for x, y in zip(X, Y)])

        N = len(contour)
        dxy = np.array([contour[(i+1) % N] - contour[i]for i in range(N)])
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0, ]), np.cumsum(dt)])
        T = t[-1]

        # compute coefficient G(0)
        a0 = 0
        c0 = 0
        for i in range(len(contour)):
            s = (t[i+1]**2 - t[i]**2)/(2*dt[i]) - t[i]
            a0 += s*dxy[i, 0] + dt[i]*(X[i]-X[0])
            c0 += s*dxy[i, 1] + dt[i]*(Y[i]-Y[0])
        FD[0] = np.complex(X[0] + T**-1*a0, Y[0] + T**-1*c0)

        # compute remaining coefficients
        for m in range(1, harmonics+1):
            omega0 = (2*np.pi*m)/T * np.array([t[i]
                                               for i in range(len(contour))])  # t
            omega1 = (
                2*np.pi*m)/T * np.array([t[(i+1) % len(contour)] for i in range(len(contour))])

            a_m = np.sum((np.cos(omega1) - np.cos(omega0)) / dt * dxy[:, 0])
            c_m = np.sum((np.cos(omega1) - np.cos(omega0)) / dt * dxy[:, 1])

            b_m = np.sum((np.sin(omega1) - np.sin(omega0)) / dt * dxy[:, 0])
            d_m = np.sum((np.sin(omega1) - np.sin(omega0)) / dt * dxy[:, 1])

            const = T/(2*np.pi*m)**2
            FD[m] = const * np.complex(a_m+d_m, c_m - b_m)
            FD[-m % new_vector_length] = const * \
                np.complex(a_m - d_m, c_m + b_m)

        return FD