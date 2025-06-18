import tifffile
import numpy as np
from scipy.optimize import curve_fit

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """
    2D Gaussian function.
    xy: tuple (x_coords, y_coords)
    amplitude: peak intensity
    xo, yo: center coordinates
    sigma_x, sigma_y: standard deviations in x and y
    offset: background level
    """
    x, y = xy
    a = 1 / (2 * sigma_x**2)
    b = 1 / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(-(a * (x - xo)**2 + b * (y - yo)**2))
    return g.ravel() # Flatten the 2D array for curve_fit

def calculate_fwhm_2d_gaussian(image_path, pixel_size_meters):
    """
    Calculates the FWHM in x and y directions by fitting a 2D Gaussian
    to the entire image.
    pixel_size_meters: The size of one pixel in meters.
    """
    try:
        # 1. Read the TIFF image
        img = tifffile.imread(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

    # Ensure it's a 2D image (grayscale)
    if img.ndim == 3:
        # If it's a color image, convert to grayscale (e.g., average channels)
        img = np.mean(img, axis=-1)
    elif img.ndim != 2:
        print(f"Error: Image must be 2D or 3D (for color). Found {img.ndim} dimensions.")
        return None

    print(f"Image loaded: {image_path}")
    print(f"Image shape: {img.shape}")
    print(f"Pixel size: {pixel_size_meters * 1e6:.3f} um") # Display pixel size in um

    # Create meshgrid for x and y coordinates
    ny, nx = img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    # Initial guess for parameters
    # Amplitude: Max pixel value
    # xo, yo: Center of the image (or peak location)
    # sigma_x, sigma_y: A reasonable guess, e.g., image_size / 6
    # offset: Min pixel value
    max_val = np.max(img)
    min_val = np.min(img)
    initial_amplitude = max_val - min_val
    initial_offset = min_val

    # Find the approximate peak location for better initial guess
    peak_y, peak_x = np.unravel_index(np.argmax(img), img.shape)

    initial_sigma_x = nx / 10 # Rough guess
    initial_sigma_y = ny / 10 # Rough guess

    initial_guess = (initial_amplitude, peak_x, peak_y, initial_sigma_x, initial_sigma_y, initial_offset)

    # Set bounds for parameters (optional, but good for stability)
    # amplitude, xo, yo, sigma_x, sigma_y, offset
    lower_bounds = [0, 0, 0, 0.1, 0.1, 0] # Sigmas must be positive
    upper_bounds = [np.inf, nx, ny, nx, ny, max_val]

    try:
        # Perform the 2D Gaussian fit
        # We need to flatten X, Y, and img for curve_fit
        popt, pcov = curve_fit(gaussian_2d, (X.ravel(), Y.ravel()), img.ravel(),
                               p0=initial_guess, bounds=(lower_bounds, upper_bounds))

        # Extract fitted parameters
        amplitude_fit, xo_fit, yo_fit, sigma_x_fit, sigma_y_fit, offset_fit = popt

        # Calculate FWHM from fitted sigmas (in pixels)
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        fwhm_x_pixels = fwhm_factor * sigma_x_fit
        fwhm_y_pixels = fwhm_factor * sigma_y_fit

        # Convert FWHM from pixels to micrometers
        fwhm_x_um = fwhm_x_pixels * pixel_size_meters * 1e6
        fwhm_y_um = fwhm_y_pixels * pixel_size_meters * 1e6

        print("\n--- 2D Gaussian Fit Results ---")
        print(f"Fitted Amplitude: {amplitude_fit:.2f}")
        print(f"Fitted Center (xo, yo): ({xo_fit:.2f}, {yo_fit:.2f}) pixels")
        print(f"Fitted Sigma X: {sigma_x_fit:.2f} pixels")
        print(f"Fitted Sigma Y: {sigma_y_fit:.2f} pixels")
        print(f"Fitted Offset (background): {offset_fit:.2f}")
        print(f"Calculated FWHM X: {fwhm_x_pixels:.2f} pixels ({fwhm_x_um:.3f} um)")
        print(f"Calculated FWHM Y: {fwhm_y_pixels:.2f} pixels ({fwhm_y_um:.3f} um)")


        # You can return a dictionary with results
        return {
            'amplitude': amplitude_fit,
            'center_x': xo_fit,
            'center_y': yo_fit,
            'sigma_x_pixels': sigma_x_fit,
            'sigma_y_pixels': sigma_y_fit,
            'offset': offset_fit,
            'fwhm_x_pixels': fwhm_x_pixels,
            'fwhm_y_pixels': fwhm_y_pixels,
            'fwhm_x_um': fwhm_x_um,
            'fwhm_y_um': fwhm_y_um
        }

    except RuntimeError as e:
        print(f"2D Gaussian fit failed: {e}. The image might not be sufficiently Gaussian or initial guess was poor.")
        return None
    except ValueError as e:
        print(f"Error during fitting: {e}. Check initial guess or bounds.")
        return None

if __name__ == "__main__":
    image_path = "beam_axicon/20250618-141802/beam_sections/intensity_z_0750.tiff"
    # Example pixel size: 5.2 micrometers (5.2e-6 meters)
    pixel_size_meters = 0.25e-6
    results_focused = calculate_fwhm_2d_gaussian(image_path, pixel_size_meters)
    if results_focused:
        print("\n--- Summary of Focused Beam Results ---")
        print(f"FWHM X: {results_focused['fwhm_x_pixels']:.2f} pixels ({results_focused['fwhm_x_um']:.3f} um)")
        print(f"FWHM Y: {results_focused['fwhm_y_pixels']:.2f} pixels ({results_focused['fwhm_y_um']:.3f} um)")