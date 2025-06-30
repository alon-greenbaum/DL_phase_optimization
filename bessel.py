import numpy as np
import matplotlib.pyplot as plt

def generate_axicon_phase_mask(
    mask_resolution_pixels=(512, 512),
    pixel_pitch_um=10,  # Micrometers per pixel on the SLM
    wavelength_nm=632.8,  # Wavelength of light (e.g., HeNe laser)
    bessel_cone_angle_degrees=1.0, # Desired cone angle of the Bessel beam in degrees
):
    """
    Generates a 2D axicon phase mask for a phase SLM to produce a Bessel beam.

    Args:
        mask_resolution_pixels (tuple): (height, width) of the phase mask in pixels.
        pixel_pitch_um (float): The physical size of each pixel on the SLM in micrometers.
        wavelength_nm (float): The wavelength of the incident light in nanometers.
        bessel_cone_angle_degrees (float): The desired cone angle of the Bessel beam
                                            in degrees. This directly relates to the
                                            slope of the axicon phase.

    Returns:
        numpy.ndarray: A 2D array representing the phase mask (values in radians).
    """

    height_pixels, width_pixels = mask_resolution_pixels

    # Convert units to a consistent system (e.g., meters)
    pixel_pitch_m = pixel_pitch_um * 1e-6  # micrometers to meters
    wavelength_m = wavelength_nm * 1e-9  # nanometers to meters
    bessel_cone_angle_rad = np.deg2rad(bessel_cone_angle_degrees) # degrees to radians

    # Create a coordinate system for the mask
    # Centered at (0,0)
    x = np.arange(-width_pixels / 2, width_pixels / 2) * pixel_pitch_m
    y = np.arange(-height_pixels / 2, height_pixels / 2) * pixel_pitch_m
    X, Y = np.meshgrid(x, y)

    # Calculate the radial coordinate
    R = np.sqrt(X**2 + Y**2)

    # Calculate the axicon constant (alpha)
    # This relates the cone angle to the phase ramp
    # For a Bessel beam, the k_r (radial wavevector) is k * sin(theta)
    # The phase mask needs to provide this radial wavevector.
    # The phase gradient for an axicon is 2*pi / L where L is the period.
    # k_r = 2*pi / L
    # We want k_r = k * sin(theta)
    # So 2*pi / L = (2*pi / wavelength) * sin(theta)
    # L = wavelength / sin(theta)
    # The phase ramp is 2*pi * R / L
    # Phase = 2*pi * R * sin(theta) / wavelength
    axicon_constant = (2 * np.pi * np.sin(bessel_cone_angle_rad)) / wavelength_m

    # Generate the phase mask
    # The phase is modulo 2*pi for an SLM (0 to 2*pi)
    phase_mask = (axicon_constant * R) % (2 * np.pi)
    phase_mask = 2 * np.pi - phase_mask  # Invert phase for SLM compatibility

    return phase_mask

def visualize_phase_mask(phase_mask, title="Axicon Phase Mask"):
    """
    Visualizes the generated phase mask.

    Args:
        phase_mask (numpy.ndarray): The 2D phase mask array.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(phase_mask, cmap='hsv', origin='lower', extent=[-1, 1, -1, 1])
    plt.colorbar(label='Phase (radians)')
    plt.title(title)
    plt.xlabel('Normalized X-coordinate')
    plt.ylabel('Normalized Y-coordinate')
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # SLM parameters (adjust these to match your SLM)
    slm_resolution = (500, 500) # Example: Full HD SLM (height, width)
    slm_pixel_pitch_um = 9.2     # Example: 8 um pixel pitch

    # Laser parameters
    laser_wavelength_nm = 561.0  # Example: Green laser

    # Desired Bessel beam properties
    desired_bessel_cone_angle_deg = 1 # A small angle for a long Bessel beam

    print(f"Generating axicon phase mask with parameters:")
    print(f"  SLM Resolution: {slm_resolution}")
    print(f"  SLM Pixel Pitch: {slm_pixel_pitch_um} um")
    print(f"  Laser Wavelength: {laser_wavelength_nm} nm")
    print(f"  Desired Bessel Cone Angle: {desired_bessel_cone_angle_deg} degrees")

    axicon_mask = generate_axicon_phase_mask(
        mask_resolution_pixels=slm_resolution,
        pixel_pitch_um=slm_pixel_pitch_um,
        wavelength_nm=laser_wavelength_nm,
        bessel_cone_angle_degrees=desired_bessel_cone_angle_deg
    )

    print(f"\nGenerated phase mask shape: {axicon_mask.shape}")
    print(f"Phase values range from {axicon_mask.min():.2f} to {axicon_mask.max():.2f} radians")

    # Visualize the generated phase mask
    visualize_phase_mask(axicon_mask)

    # You would typically load this `axicon_mask` array onto your SLM using its
    # specific SDK or API. This often involves converting the phase (0 to 2pi)
    # to the SLM's grayscale values (e.g., 0-255 for an 8-bit SLM).

    # Example of how to convert to 8-bit grayscale for an SLM (if needed):
    # slm_grayscale_mask = (axicon_mask / (2 * np.pi) * 255).astype(np.uint8)
    # plt.figure(figsize=(8, 7))
    # plt.imshow(slm_grayscale_mask, cmap='gray', origin='lower')
    # plt.colorbar(label='Grayscale Value (0-255)')
    # plt.title('Axicon Phase Mask (8-bit grayscale for SLM)')
    # plt.show()