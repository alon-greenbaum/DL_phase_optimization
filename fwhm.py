import tifffile
import numpy as np
from scipy.interpolate import interp1d

def calculate_fwhm(image_path):
    """
    Calculates the FWHM of a Gaussian distribution in a TIFF image.
    Assumes the Gaussian is roughly centered and finds FWHM along the
    row with the maximum intensity.
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

    # Find the row and column of the maximum intensity
    max_intensity_idx = np.unravel_index(np.argmax(img), img.shape)
    peak_row = max_intensity_idx[0]
    peak_col = max_intensity_idx[1]

    # 2. Extract a 1D profile (cross-section through the peak)
    # We'll take the row profile for now. You could also do a column profile
    # and average the FWHMs, or fit a 2D Gaussian for more accuracy.
    profile = img[peak_row, :]

    # 3. Find the maximum intensity
    max_val = np.max(profile)

    # 4. Calculate half-maximum
    half_max = max_val / 2.0

    # Subtract baseline if necessary (e.g., if there's a constant offset)
    # For a perfect Gaussian, the minimum should be close to zero.
    # If your image has a significant background, consider subtracting it:
    # baseline = np.min(profile)
    # profile_no_baseline = profile - baseline
    # max_val_no_baseline = np.max(profile_no_baseline)
    # half_max_no_baseline = max_val_no_baseline / 2.0 + baseline # Add baseline back to get absolute intensity for crossing points

    # 5. Find the points at half-maximum using interpolation for accuracy
    # Create an x-axis for the profile
    x_coords = np.arange(len(profile))

    # Find where the profile crosses the half-maximum value
    # We'll use interpolation to get a more precise crossing point
    # rather than just pixel indices.

    # Find the indices where the intensity is greater than or equal to half_max
    indices_above_half_max = np.where(profile >= half_max)[0]

    if len(indices_above_half_max) < 2:
        print("Could not find enough points above half maximum to calculate FWHM.")
        return None

    # Get the leftmost and rightmost points where the intensity crosses half_max
    # We'll interpolate for better precision
    try:
        # Left side: Find the first point where intensity rises above half_max
        # Iterate from the center outwards to find the specific crossing points.
        # This is more robust for non-ideal Gaussian profiles.

        # Find the index closest to the peak on the left that is below half_max
        left_idx = np.where(profile[:peak_col] < half_max)[0]
        if len(left_idx) == 0: # If the left side starts above half_max
            x1 = x_coords[0]
        else:
            left_idx = left_idx[-1] # Last point below half_max before rising
            # Interpolate between this point and the next one (which is >= half_max)
            f_left = interp1d(profile[left_idx:left_idx+2], x_coords[left_idx:left_idx+2])
            x1 = f_left(half_max)

        # Right side: Find the first point where intensity falls below half_max
        right_idx = np.where(profile[peak_col:] < half_max)[0]
        if len(right_idx) == 0: # If the right side ends above half_max
            x2 = x_coords[-1]
        else:
            right_idx = right_idx[0] + peak_col # First point below half_max after falling
            # Interpolate between this point and the previous one (which is >= half_max)
            f_right = interp1d(profile[right_idx-1:right_idx+1][::-1], x_coords[right_idx-1:right_idx+1][::-1])
            x2 = f_right(half_max)

    except ValueError:
        print("Could not interpolate FWHM crossing points. The profile might not be a clear Gaussian.")
        return None

    # 6. Calculate the distance (FWHM)
    fwhm = x2 - x1

    print(f"Image loaded: {image_path}")
    print(f"Image shape: {img.shape}")
    print(f"Peak intensity located at pixel ({peak_row}, {peak_col})")
    print(f"Maximum intensity: {max_val:.2f}")
    print(f"Half maximum intensity: {half_max:.2f}")
    print(f"Calculated FWHM (along row {peak_row}): {fwhm:.2f} pixels")

    return fwhm

if __name__ == "__main__":
    # Example usage:
    # Create a dummy TIFF image with a Gaussian for testing
    img_size = 200
    x = np.linspace(-10, 10, img_size)
    y = np.linspace(-10, 10, img_size)
    X, Y = np.meshgrid(x, y)
    sigma_x = 2.5
    sigma_y = 3.0 # Slightly elliptical Gaussian
    amplitude = 255
    gaussian_img = amplitude * np.exp(-((X**2 / (2 * sigma_x**2)) + (Y**2 / (2 * sigma_y**2))))

    # Save the dummy image as a TIFF file
    dummy_image_path = "gaussian_test_image.tif"
    tifffile.imwrite(dummy_image_path, gaussian_img.astype(np.uint8)) # Save as 8-bit for simplicity

    image_path = "/home/regarry/dd_phase_optimization/beam_axicon/20250612-130812/beam_sections/intensity_z_0600.tiff"
    # Now, calculate FWHM for the dummy image
    fwhm_result = calculate_fwhm(image_path)

    # You can also test with a non-existent path
    # calculate_fwhm("non_existent_image.tif")