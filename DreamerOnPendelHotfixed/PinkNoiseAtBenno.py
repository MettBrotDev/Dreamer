import numpy as np
from numpy.fft import irfft, rfftfreq


class PinkNoise:
    def __init__(self, size, scale=1, max_period=None, rng=None):
        """
        Parameters:
        -----------
        size : tuple (sequence_length, action_dim)
            Defines the number of actions and time steps.
        scale : float, optional
            Scaling factor for noise amplitude (default=1).
        max_period : float, optional
            Defines the longest correlation length (default: size[-1]).
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.sequence_length, self.action_dim = size
        self.minimum_frequency = 1 / max_period if max_period else 0
        self.scale = scale
        self.rng = rng if rng else np.random.default_rng(0)

        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[0]

        self.reset()

    def __call__(self):
        """
        Retrieve the next sample from the precomputed pink noise buffer.

        Returns:
        --------
        np.ndarray:
            A sample from the generated pink noise.
        """
        if self.time_steps >= self.sequence_length:
            self.reset()

        noise_sample = self.buffer[:, self.time_steps]
        self.time_steps += 1

        std_dev = np.std(noise_sample)
        if std_dev == 0:
            return self.scale * noise_sample
        return self.scale * noise_sample / std_dev

    def generate_pink_noise(self, size, fmin=0, rng=None):
        """
        Generate pink noise with the given size and minimum frequency.

        Parameters:
        -----------
        size : tuple
            The size of the noise array.
        minimum_frequency : float
            The minimum frequency of the noise.
        rng : np.random.Generator
            Random number generator for reproducibility.

        Returns:
        --------
        np.ndarray
            A pink noise array of the given size.
        """
        try:
            size = list(size)
        except TypeError:
            size = [size]

        num_samples = size[-1]
        freqs = rfftfreq(num_samples)

        # Validate / normalize fmin
        if 0 <= fmin <= 0.5:
            fmin = max(fmin, 1./num_samples) # Low frequency cutoff
        else:
            raise ValueError("fmin must be chosen between 0 and 0.5.")

        scaling_factors = np.where(freqs < fmin, fmin, freqs) ** (-1/2.) # Build scaling factors for all frequencies

        w = scaling_factors[1:].copy() # generate white noise as base
        if num_samples % 2 == 0:
            w[-1] *= 1.5  # Correct for Nyquist frequency
        sigma = 2 * np.sqrt(np.sum(w**2)) / num_samples

        size[-1] = len(freqs) # Adjust size for Fourier components
        scaling_factors = np.reshape(scaling_factors, (1,) * (len(size) - 1) + (-1,))

        rng = rng or np.random.default_rng()
        real_part = rng.normal(scale=scaling_factors, size=size)
        imag_part = rng.normal(scale=scaling_factors, size=size)

        # Ensure correct behavior at DC and Nyquist frequency
        real_part[..., 0] *= np.sqrt(2)
        imag_part[..., 0] = 0  # DC component is real
        if num_samples % 2 == 0:
            real_part[..., -1] *= np.sqrt(2)
            imag_part[..., -1] = 0  # Nyquist component is real


        spectrum = real_part + 1j * imag_part # Combine real and imaginary parts

        # Perform inverse FFT and normalize variance
        pink_noise = irfft(spectrum, n=num_samples, axis=-1) / sigma

        return pink_noise

    def reset(self):
        """
        Resets the pink noise buffer.
        """
        self.buffer = self.generate_pink_noise(self.size, self.minimum_frequency, self.rng)
        self.time_steps = 0