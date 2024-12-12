# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.integrate
import scipy.signal
import scipy.special
import scipy.io as sio

N = 5000  # grid size
px = 1e-6  # pixel size (um)
focal_length = 55e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6


## this part defines the "blur" of the beads at different Z sections, it is used in mask optimization part
def get_airy_psf(psf_width_pixels, psf_width_meters, z, wavelength, numerical_aperture, refractive_index,
                 normalize=True):
    """
    psf_width_pixels: Integer, the width of the psf, in pixels. Must be odd. If this is even, testGetAiryPsfGoldenZeroDepth() will fail.
    psf_width_meters: Float, the width of the psf, in meters.
    z: Float, z-coordinate relative to the focal plane, in meters.

    Returns:
        psf kernel
    """
    meters_per_pixel = psf_width_meters / psf_width_pixels
    psf = np.zeros((psf_width_pixels, psf_width_pixels), dtype=np.float64)
    for i in range(psf_width_pixels):
        for j in range(psf_width_pixels):
            x = (i - (psf_width_pixels - 1.0) / 2.0) * meters_per_pixel
            y = (j - (psf_width_pixels - 1.0) / 2.0) * meters_per_pixel
            if (i - (psf_width_pixels - 1.0) / 2.0) ** 2 + (j - (psf_width_pixels - 1.0) / 2.0) <= (
                    (psf_width_pixels - 1.0) / 2.0) ** 2:
                psf[i, j] = eval_airy_function(x, y, z, wavelength, numerical_aperture, refractive_index)

    # Normalize PSF to max value.
    if normalize:
        return psf / np.max(psf)
    return psf


def eval_airy_function(x, y, z, wavelength, NA, refractive_index):
    """
    Parameters
    ----------
    x: Float, x coordinate, in meters.
    y: Float, y coordinate, in meters.
    z: Float, z coordinate, in meters.
    wavelength: Float, wavelength of light in meters.
    numerical_aperture: Float, numerical aperture of the imaging lens.
    refractive_index: Float, refractive index of the imaging medium.

    Returns
    -------
    values of psf
    """
    k = 2 * np.pi / wavelength
    n = refractive_index

    def fun_integrate(rho):
        bessel_arg = k * NA / n * np.sqrt(x * x + y * y) * rho
        return scipy.special.j0(bessel_arg) * np.exp(-0.5 * 1j * k * rho * rho * z * np.power(NA / n, 2)) * rho

    integral_result = integrate_numerical(fun_integrate, 0.0, 1.0)

    return float(np.real(integral_result * np.conj(integral_result)))


def integrate_numerical(function_to_integrate, start, end):
    def real_function(x):
        return np.real(function_to_integrate(x))

    def imag_function(x):
        return np.imag(function_to_integrate(x))

    real_result = scipy.integrate.quad(real_function, start, end)[0]
    imag_result = scipy.integrate.quad(imag_function, start, end)[0]
    return real_result + 1j * imag_result


def apply_blur_kernel(image, psf):
    psf_normalized = psf / np.sum(psf)
    return scipy.signal.convolve2d(image, psf_normalized, 'same', boundary='symm')


def generate_psf():
    z_depth = np.arange(0, 4.1e-5, 1e-6)
    psf = []
    for i in range(41):
        print(i)
        psf.append(get_airy_psf(psf_width_pixels, psf_width_meters, z_depth[i], wavelength, numerical_aperture,
                                refractive_index))
    sio.savemat('psf_z.mat', {'psf': psf})

    return psf


if __name__ == '__main__':
    generate_psf()
