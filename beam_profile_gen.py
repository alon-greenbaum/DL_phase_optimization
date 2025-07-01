import os
import numpy as np
import skimage.io

class BeamProfiler:
    """
    A class to simulate beam propagation through optical elements.
    
    This class encapsulates the parameters and methods for simulating
    the propagation of a light beam, including phase mask generation,
    propagation to a focal point, and propagation using the angular
    spectrum method.
    """

    def __init__(self, config: dict):
        """
        Initializes the BeamSimulator with parameters from a config dictionary.
        
        Args:
            config (dict): A dictionary containing all simulation parameters.
        """
        # Unpack the configuration dictionary into class attributes
        for key, value in config.items():
            setattr(self, key, value)
        
        self._create_coordinate_grids()

    def _create_coordinate_grids(self):
        """
        Pre-calculates coordinate grids to avoid redundant computations.
        These grids represent the physical dimensions of the simulation space.
        """
        # Primary grid
        xy_range = np.arange(-self.N // 2, self.N // 2)
        X, Y = np.meshgrid(xy_range, xy_range)
        self.X_m = X * self.px  # Grid coordinates in meters
        self.Y_m = Y * self.px  # Grid coordinates in meters

        # Padded grid for Fresnel propagation
        xy_range_padded = np.arange(-self.N + 1, self.N + 1)
        XX_padded, YY_padded = np.meshgrid(xy_range_padded, xy_range_padded)
        self.XX_m_padded = XX_padded * self.px
        self.YY_m_padded = YY_padded * self.px

    @staticmethod
    def normalize_to_uint16(img: np.ndarray) -> np.ndarray:
        """
        Normalizes a numpy array to the uint16 range [0, 65535].
        
        Args:
            img (np.ndarray): The input image array.
            
        Returns:
            np.ndarray: The normalized image as a uint16 array.
        """
        img_float = img.astype(np.float32)
        img_float -= img_float.min()
        max_val = img_float.max()
        if max_val > 0:
            img_float /= max_val
        return (img_float * 65535).astype(np.uint16)

    def generate_gaussian_mask(self, fwhm: float = 0.0001) -> np.ndarray:
        """
        Generates a Gaussian amplitude mask.
        
        Args:
            fwhm (float): Full Width at Half Maximum of the Gaussian profile.
            
        Returns:
            np.ndarray: The generated Gaussian mask.
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-(np.square(self.X_m) + np.square(self.Y_m)) / (2 * sigma**2))

    def propagate_to_focus(self, complex_mask: np.ndarray) -> np.ndarray:
        """
        Propagates a masked beam to the focal plane of a lens using Fresnel diffraction.
        
        Args:
            complex_mask (np.ndarray): The complex field at the initial plane (mask).
            
        Returns:
            np.ndarray: The complex field at the focal plane.
        """
        # Phase transformation of the lens
        lens_phase = (np.pi / (self.wavelength * self.focal_length)) * (np.square(self.X_m) + np.square(self.Y_m))
        incident_field = complex_mask * np.exp(-1j * lens_phase)

        # Fresnel propagation kernel
        fresnel_kernel = np.exp(1j * (np.pi / (self.wavelength * self.focal_length)) * (np.square(self.XX_m_padded) + np.square(self.YY_m_padded)))
        
        # Perform propagation using the convolution theorem
        padded_field = np.pad(incident_field, self.N // 2)
        propagated_field_fft = np.fft.fft2(padded_field) * np.fft.fft2(fresnel_kernel)
        propagated_field = np.fft.ifftshift(np.fft.ifft2(propagated_field_fft))

        # Crop the result to the original size
        return propagated_field[self.N // 2:3 * self.N // 2, self.N // 2:3 * self.N // 2]

    def angular_spectrum_propagation(self, field: np.ndarray, z: float) -> np.ndarray:
        """
        Propagates a field over a distance z using the angular spectrum method.
        
        Args:
            field (np.ndarray): The initial complex field.
            z (float): The propagation distance in meters.
            
        Returns:
            np.ndarray: The intensity (real-valued) of the propagated field.
        """
        k = 2 * np.pi / self.wavelength
        
        # Spatial frequency coordinates
        fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.px))
        fy = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.px))
        Fx, Fy = np.meshgrid(fx, fy)

        # Wave number components
        alpha = self.wavelength * Fx
        beta = self.wavelength * Fy
        
        # The argument of the square root must be non-negative
        gamma_sq = 1 - np.square(alpha) - np.square(beta)
        gamma = np.sqrt(np.maximum(gamma_sq, 0))

        # Transfer function for propagation
        transfer_function = np.exp(1j * k * gamma * z)
        
        field_fft = np.fft.fftshift(np.fft.fft2(field))
        propagated_field_fft = field_fft * transfer_function
        propagated_field = np.fft.ifft2(np.fft.ifftshift(propagated_field_fft))

        # Return the intensity (magnitude squared)
        return np.abs(propagated_field)**2

    def generate_beam_cross_section(self, initial_field: np.ndarray, output_folder: str, z_range_px: tuple, y_slice_px: tuple) -> np.ndarray:
        """
        Generates a 2D cross-section of the beam by stacking 1D slices along the z-axis.
        
        Args:
            initial_field (np.ndarray): The complex field at z=0.
            output_folder (str): Folder to save 2D intensity profiles at each z-step.
            z_range_px (tuple): (min, max) propagation distance in pixels.
            y_slice_px (tuple): (min, max) slice of the y-axis in pixels.
            
        Returns:
            np.ndarray: A 2D array representing the beam's cross-section (ZY plane).
        """
        os.makedirs(output_folder, exist_ok=True)
        z_min_px, z_max_px = z_range_px
        y_min_px, y_max_px = y_slice_px
        
        num_z_steps = z_max_px - z_min_px
        cross_section_profile = np.zeros((num_z_steps, y_max_px - y_min_px))

        print(f"Generating beam cross-section for {num_z_steps} slices...")
        for i, z_step_px in enumerate(range(z_min_px, z_max_px)):
            z_dist_m = z_step_px * self.px
            
            # Propagate field and get intensity
            intensity_at_z = self.angular_spectrum_propagation(initial_field, z_dist_m)

            # Extract the central 1D slice along the y-axis
            center_row_idx = intensity_at_z.shape[0] // 2
            center_col_idx = intensity_at_z.shape[1] // 2
            cross_section_profile[i,:] = intensity_at_z[center_row_idx, center_col_idx + y_min_px : center_col_idx + y_max_px]

            # Save the full 2D intensity profile at the current z-step
            save_path = os.path.join(output_folder, f'intensity_z_{i:04d}.tiff')
            skimage.io.imsave(save_path, self.normalize_to_uint16(intensity_at_z))
            
        print("Cross-section generation complete.")
        return self.normalize_to_uint16(cross_section_profile)


if __name__ == '__main__':
    # Define all simulation parameters in a configuration dictionary
    config = {
        'N': 500,                      # grid size
        'px': 1e-6,                    # pixel size (m)
        'focal_length': 2e-3,          # focal length (m)
        'wavelength': 0.561e-6,        # wavelength (m)
        'refractive_index': 1.0,
    }

    # 1. Initialize the simulator with the configuration
    simulator = BeamSimulator(config)

    # 2. Create the initial complex mask
    # Amplitude part (e.g., a Gaussian mask)
    amplitude_mask = simulator.generate_gaussian_mask(fwhm=0.0001)
    
    # Phase part (e.g., a simple linear tilt)
    phase_values = np.linspace(0, 2 * np.pi, config['N'])
    phase_mask = np.tile(phase_values, (config['N'], 1)) # Creates a horizontal phase ramp
    
    # Combine amplitude and phase into a complex mask
    complex_mask = amplitude_mask * np.exp(1j * phase_mask)

    # 3. Propagate the beam to the focal plane to get the initial field for 3D propagation
    field_at_focus = simulator.propagate_to_focus(complex_mask)
    skimage.io.imsave('field_at_focus_intensity.tiff', simulator.normalize_to_uint16(np.abs(field_at_focus)**2))

    # 4. Generate a 3D beam profile and save its cross-section
    # Define the range for propagation (z-axis) and slicing (y-axis) in pixels
    z_range_pixels = (0, 200)       # Propagate from z=0 to z=200 pixels (in steps of 1 pixel)
    y_slice_pixels = (-50, 50)      # Slice the central 100 pixels along the y-axis

    # Generate the cross-section (ZY plane)
    beam_profile_zy = simulator.generate_beam_cross_section(
        initial_field=field_at_focus,
        output_folder='beam_cross_sections',
        z_range_px=z_range_pixels,
        y_slice_px=y_slice_pixels
    )

    # 5. Save the final cross-section image
    skimage.io.imsave('final_beam_cross_section_ZY.tiff', beam_profile_zy)
    print("Simulation finished. Final cross-section saved to 'final_beam_cross_section_ZY.tiff'")