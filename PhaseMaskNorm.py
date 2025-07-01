import numpy as np
from PIL import Image
import math
import os

def normalize_phase_image(input_tiff_path, output_bmp_path, slm_width=None, slm_height=None, padding_mode=False):
    """
    Loads a TIFF image representing a phase mask, normalizes its phase values
    to the [0, 2*pi) range, scales them to [0, 255], and saves as an 8-bit BMP.
    Optionally resizes or pads the phase mask to match the SLM's dimensions.

    Args:
        input_tiff_path (str): Path to the input TIFF phase mask image.
        output_bmp_path (str): Path to save the output 8-bit BMP image.
        slm_width (int, optional): Desired width of the output image in pixels,
                                    matching the SLM's resolution. If None, original width is used.
        slm_height (int, optional): Desired height of the output image in pixels,
                                     matching the SLM's resolution. If None, original height is used.
        padding_mode (bool): If True and slm_width/height are provided, the image will be padded
                             to the target dimensions instead of resized. If False, it will be resized.
    """
    try:
        # 1. Load the TIFF image
        print(f"Loading TIFF image from: {input_tiff_path}")
        img = Image.open(input_tiff_path)

        # Convert to a NumPy array for numerical operations
        initial_phase_array = np.array(img, dtype=np.float64)
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width}x{original_height} pixels")
        print(f"Original image data type: {initial_phase_array.dtype}")
        print(f"Original phase array min: {initial_phase_array.min()}, max: {initial_phase_array.max()}")

        # Define P (2*pi for phase normalization)
        P = 2 * math.pi

        # 2. Apply the normalization formula: (initial_phase % P + P) % P
        print(f"Normalizing phase to the [0, {P:.4f}) range...")
        normalized_phase_array = (np.fmod(initial_phase_array, P) + P) % P
        print(f"Normalized phase array min: {normalized_phase_array.min()}, max: {normalized_phase_array.max()}")

        target_width = slm_width if slm_width is not None else original_width
        target_height = slm_height if slm_height is not None else original_height

        # 2.5. Handle resizing or padding
        if target_width != original_width or target_height != original_height:
            if padding_mode:
                print(f"Padding phase mask to SLM dimensions: {target_width}x{target_height} pixels...")
                # Create a new array filled with 0.0 (representing 0 phase)
                padded_array = np.zeros((target_height, target_width), dtype=np.float64)

                # Calculate start coordinates for pasting the original image
                start_x = (target_width - original_width) // 2
                start_y = (target_height - original_height) // 2

                # Ensure dimensions fit
                if start_x < 0 or start_y < 0:
                    raise ValueError("Target SLM dimensions are smaller than the original image. Cannot pad, consider resizing.")

                # Place the normalized original image into the center of the padded array
                padded_array[start_y:start_y + original_height, start_x:start_x + original_width] = normalized_phase_array
                current_phase_array = padded_array
                print(f"Padded phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")

            else: # Resizing mode (default if padding_mode is False)
                print(f"Resizing phase mask to SLM dimensions: {target_width}x{target_height} pixels...")
                temp_img = Image.fromarray(normalized_phase_array, mode='F') # Use normalized array for resizing
                resized_img = temp_img.resize((target_width, target_height), Image.Resampling.BICUBIC)
                current_phase_array = np.array(resized_img, dtype=np.float64)
                print(f"Resized phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")
        else:
            current_phase_array = normalized_phase_array


        # 3. Scale the (potentially resized/padded) normalized phase from [0, 2*pi) to [0, 255] for 8-bit BMP
        if P == 0:
            raise ValueError("Period P (2*pi) cannot be zero for scaling.")

        print("Scaling normalized phase to 0-255 range for 8-bit image...")
        scaled_image_array = (current_phase_array / P) * 255.0
        
        # Round to nearest integer and ensure values are within 0-255
        scaled_image_array = np.round(scaled_image_array).astype(np.uint8)
        print(f"Scaled image array min: {scaled_image_array.min()}, max: {scaled_image_array.max()}")

        # 4. Save the resulting image as an 8-bit BMP
        output_img = Image.fromarray(scaled_image_array, mode='L') # 'L' for 8-bit grayscale
        
        print(f"Saving 8-bit BMP image to: {output_bmp_path}")
        output_img.save(output_bmp_path)
        print("Processing complete!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_tiff_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define input and output file paths
    # IMPORTANT: Replace 'input_phase_mask.tiff' with the actual path to your TIFF file.
    # The output 'output_normalized_phase.bmp' will be created in the same directory
    # where you run this script, unless you specify a different path.
    
    
    """# Create a dummy TIFF for demonstration if one doesn't exist
    dummy_input_tiff = "dummy_phase_mask.tiff"
    if not os.path.exists(dummy_input_tiff):
        print(f"Creating a dummy TIFF file for demonstration: {dummy_input_tiff}")
        # Create a 500x500 image with phase values ranging from -4*pi to 4*pi
        # This closely matches the user's scenario
        dummy_data = np.linspace(-4 * math.pi, 4 * math.pi, 500 * 500).reshape((500, 500))
        dummy_data += np.sin(np.arange(500)/25.0)[:, np.newaxis] * math.pi / 2
        
        dummy_img = Image.fromarray(dummy_data.astype(np.float32), mode='F')
        dummy_img.save(dummy_input_tiff)
        print("Dummy TIFF created. You can now run the script.")
    else:
        print(f"Dummy TIFF file already exists: {dummy_input_tiff}")
    """
    input_file = "/home/regarry/dd_phase_optimization/training_results/phase_model_20250625-142441/mask_phase_epoch_150_499.tiff" # Use the dummy file for demonstration
    output_file = "/home/regarry/dd_phase_optimization/training_results/phase_model_20250625-142441/epoch_150_normalized_phase.bmp"


    # --- SET YOUR SLM DIMENSIONS HERE ---
    slm_target_width = 1920
    slm_target_height = 1152

    # --- CHOOSE PADDING MODE ---
    # Set padding_mode=True to pad the image.
    # Set padding_mode=False (or omit) to resize the image (previous behavior).
    use_padding = True 

    normalize_phase_image(input_file, output_file, 
                          slm_width=slm_target_width, 
                          slm_height=slm_target_height, 
                          padding_mode=use_padding)

    print(f"\nCheck '{output_file}' for the normalized and {'padded' if use_padding else 'resized'} 8-bit phase mask.")
    print("Remember to replace 'dummy_phase_mask.tiff' with your actual input file path.")
