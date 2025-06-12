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

def calculate_fwhm_2d_gaussian(image_path):
    """
    Calculates the FWHM in x and y directions by fitting a 2D Gaussian
    to the entire image.
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

        # Calculate FWHM from fitted sigmas
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        fwhm_x = fwhm_factor * sigma_x_fit
        fwhm_y = fwhm_factor * sigma_y_fit

        print("\n--- 2D Gaussian Fit Results ---")
        print(f"Fitted Amplitude: {amplitude_fit:.2f}")
        print(f"Fitted Center (xo, yo): ({xo_fit:.2f}, {yo_fit:.2f}) pixels")
        print(f"Fitted Sigma X: {sigma_x_fit:.2f} pixels")
        print(f"Fitted Sigma Y: {sigma_y_fit:.2f} pixels")
        print(f"Fitted Offset (background): {offset_fit:.2f}")
        print(f"Calculated FWHM X: {fwhm_x:.2f} pixels")
        print(f"Calculated FWHM Y: {fwhm_y:.2f} pixels")

        # You can return a dictionary with results
        return {
            'amplitude': amplitude_fit,
            'center_x': xo_fit,
            'center_y': yo_fit,
            'sigma_x': sigma_x_fit,
            'sigma_y': sigma_y_fit,
            'offset': offset_fit,
            'fwhm_x': fwhm_x,
            'fwhm_y': fwhm_y
        }

    except RuntimeError as e:
        print(f"2D Gaussian fit failed: {e}. The image might not be sufficiently Gaussian or initial guess was poor.")
        return None
    except ValueError as e:
        print(f"Error during fitting: {e}. Check initial guess or bounds.")
        return None

if __name__ == "__main__":
    image_path = "beam_axicon/20250612-140010/beam_sections/intensity_z_0099.tiff"
    results_focused = calculate_fwhm_2d_gaussian(image_path)
    if results_focused:
        print("\n--- Summary of Focused Beam Results ---")
        print(f"FWHM X: {results_focused['fwhm_x']:.2f} pixels")
        print(f"FWHM Y: {results_focused['fwhm_y']:.2f} pixels")
    """  # Example usage:
    # Create a dummy TIFF image with a 2D Gaussian for testing
    img_size = 200
    x_coords_grid = np.arange(img_size)
    y_coords_grid = np.arange(img_size)
    X_grid, Y_grid = np.meshgrid(x_coords_grid, y_coords_grid)

    # Parameters for the dummy Gaussian
    amplitude_true = 200
    xo_true = 95.5 # Slightly off center
    yo_true = 105.2 # Slightly off center
    sigma_x_true = 10.0 # Wide in X
    sigma_y_true = 5.0  # Narrow in Y (elliptical)
    offset_true = 50 # Background

    # Generate the 2D Gaussian image
    dummy_gaussian_img = offset_true + amplitude_true * np.exp(
        -(((X_grid - xo_true)**2 / (2 * sigma_x_true**2)) +
          ((Y_grid - yo_true)**2 / (2 * sigma_y_true**2)))
    )

    # Add some noise to make it more realistic
    dummy_gaussian_img += np.random.normal(0, 5, dummy_gaussian_img.shape)

    # Ensure integer values for image (e.g., 8-bit image)
    dummy_gaussian_img = np.clip(dummy_gaussian_img, 0, 255).astype(np.uint8)

    # Save the dummy image as a TIFF file
    dummy_image_path = "gaussian_2d_test_image.tif"
    tifffile.imwrite(dummy_image_path, dummy_gaussian_img)

    # Now, calculate FWHM for the dummy image using 2D fit
    results = calculate_fwhm_2d_gaussian(dummy_image_path)

    if results:
        print("\n--- Summary of Results ---")
        print(f"Average FWHM: {(results['fwhm_x'] + results['fwhm_y']) / 2:.2f} pixels")
        # Example of how you might use the individual FWHM values
        if abs(results['fwhm_x'] - results['fwhm_y']) > 1.0: # Arbitrary threshold for "elliptical"
            print("Beam appears to be elliptical.")
        else:
            print("Beam appears to be approximately circular.")

    print("\n" + "="*50 + "\n")

    # Test with a very focused beam (where 1D profile might fail)
    sigma_x_very_narrow = 1.0
    sigma_y_very_narrow = 0.8
    focused_gaussian_img = offset_true + amplitude_true * np.exp(
        -(((X_grid - xo_true)**2 / (2 * sigma_x_very_narrow**2)) +
          ((Y_grid - yo_true)**2 / (2 * sigma_y_very_narrow**2)))
    )
    focused_gaussian_img += np.random.normal(0, 5, focused_gaussian_img.shape)
    focused_gaussian_img = np.clip(focused_gaussian_img, 0, 255).astype(np.uint8)
    focused_image_path = "gaussian_focused_2d_test_image.tif"
    tifffile.imwrite(focused_image_path, focused_gaussian_img)

    print("\n--- Testing with a very focused beam (2D fit should handle it) ---")
    """
    