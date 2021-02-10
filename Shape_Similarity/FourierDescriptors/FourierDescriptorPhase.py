import numpy as np
from FourierDescriptorBase import FourierDescriptorBase


class FourierDescriptorPhase(FourierDescriptorBase):
    """
    Class to construct phase preserving Fourier descriptor of given binary images \n
    @param list of filenames or list of images
    @param number of fourier descriptor (FD) harmonics -> FD[-m_n, -m_n-1, ...-m_1, m_0, m_1, ...., m_n-1, m_n ] for n harmonics
    @return pair of Fourier descriptors G1 and G2
    """

    def normalizeDescriptor(self, descriptor: np.ndarray) -> np.ndarray:
        self._setTranslationInvariant(descriptor)
        self._setScaleInvariant(descriptor)
        G_a, G_b = self._setStartPointInvariant(descriptor)

        # rotation

        G_ac = np.concatenate( (G_a.real, G_a.imag))
        G_bc = np.concatenate( (G_b.real, G_b.imag)) 

        return np.concatenate((G_ac, G_bc))

    def _setTranslationInvariant(self, descriptor: np.ndarray):
        """
        @brief Makes given descriptor translation invariant
        @param descriptor Fourier descriptor
        """
        descriptor[0] = 0

    def _setScaleInvariant(self, descriptor: np.ndarray):
        """
        @brief Makes given descriptor scale invariant
        @param descriptor Fourier descriptor
        """
        s = 0
        for m in range(1, self.descriptor_harmonics+1):
            s += np.abs(descriptor[-m]) + np.abs(descriptor[m])
        v = 1.0/np.sqrt(s)

        for m in range(1, self.descriptor_harmonics+1):
            descriptor[-m] *= v
            descriptor[m] *= v

    def _setStartPointInvariant(self, descriptor: np.ndarray) -> np.ndarray:
        """
        @brief Make Fourier Descriptor invariant to start point phase phi and phi + np.pi
        @param descriptor Fourier descriptor
        """
        phi = self._getStartPointPhase(descriptor)
        G_a = self._shiftStartPointPhase(descriptor, phi)
        G_b = self._shiftStartPointPhase(descriptor, phi + np.pi)

        return G_a, G_b

    def _getStartPointPhase(self, descriptor: np.ndarray) -> float:
        """
        @brief  Returns start point phase phi by maximizing function _fp(descriptor,phi), with phi [0,np.pi)
                The maximum is simple brute-force search (OPTIMIZE!!)
        @param descriptor Fourier descriptor
        """
        c_max = -float("inf")
        phi_max = 0
        K = 400  # brute force with 400 steps TO DO: OPTIMIZE!!

        for k in range(K):
            phi = np.pi * float(k)/K
            c = self._fp(descriptor, phi)
            if c > c_max:
                c_max = c
                phi_max = phi

        return phi_max

    def _fp(self, descriptor: np.ndarray, phi: float):
        """
        @brief Look for quantity that depends only on the phase differneces within the Fourier descriptor pairs
        @param descriptor Fourier descriptor
        @param phi start point phase
        """
        s = 0
        for m in range(1, self.descriptor_harmonics+1):
            z1 = descriptor[-m]*np.exp(- 1j*m*phi)
            z2 = descriptor[m]*np.exp(1j*m*phi)
            s += z1.real * z2.imag - z1.imag * z2.real
        return s

    def _shiftStartPointPhase(self, descriptor: np.ndarray, phi: float) -> np.ndarray:
        """
        @brief normalizes discriptor by shifting start point phase
        @param descriptor Fourier descriptor
        @param phi start point phase
        """
        G = np.copy(descriptor)
        for m in range(1, self.descriptor_harmonics+1):
            G[-m] = descriptor[-m] * np.exp(-1j*m*phi)
            G[m] = descriptor[m] * np.exp(1j*m*phi)

        return G

    def getDescriptorDimensions(self) -> int:
        """
        @brief For n harmonics => FD[-m_n, ...., 0, ...., m_n ]
                We have to FDs (G_a, G_b)
                We separate in real part and imaginary part 
        """
        return (2*self.descriptor_harmonics+1)*2**2
