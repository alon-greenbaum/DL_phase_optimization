import os
import argparse
import numpy as np
import torch
import skimage.io
import matplotlib.pyplot as plt
from datetime import datetime
from data_utils import load_config, normalize_to_uint16
#from beam_profile_gen import BeamProfiler
from bessel import generate_axicon_phase_mask
from physics_utils import PhysicalLayer

def main():
    parser = argparse.ArgumentParser(description="Test light propagation with optional axicon phase mask and save beam profile.")
    parser.add_argument("--config", type=str, required=False, help="Path to config.yaml")
    parser.add_argument("--mask", type=str, default="", help="Optional: path to phase mask tiff (default: zeros)")
    parser.add_argument("--output_dir", type=str, default="beam_profile_test", help="Output directory")
    parser.add_argument("--fresnel_lens_pattern", action="store_true", help="Use Fresnel lens phase mask")
    args = parser.parse_args()

    # Create a timestamped subfolder output dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_subdir, exist_ok=True)
    if args.mask and not args.config:
        print(f"Using provided config and mask: {args.mask}")
        mask_config_path = os.path.join(os.path.dirname(args.mask), "config.yaml")
        config = load_config(mask_config_path)
    else:
        config = load_config(args.config)
    N = config['N']
    px = float(config['px'])  # pixel size in meters
    config['px'] = px  # Store pixel size in config for later use
    px_mm = px * 1e3 # px in mm
    px_um = px * 1e6 # px in um
    wavelength_nm = config['wavelength'] * 1e9 # wavelength in nm
    beam_fwhm = config['laser_beam_FWHC']
    bessel_angle = config['bessel_cone_angle_degrees']
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config['device']
    asm = config.get('angular_spectrum_method', True)
    initial_phase_mask = config.get('initial_phase_mask', "")

    # Get z_min, z_max, y_min, y_max, and num_z_steps from config
    z_min_mm = config.get('z_min_mm', -10.0) # Default to -10 mm if not in config
    z_max_mm = config.get('z_max_mm', 10.0)  # Default to 10 mm if not in config
    y_min_mm = config.get('y_min_mm', -1.0)  # Default to -1 mm if not in config
    y_max_mm = config.get('y_max_mm', 1.0)   # Default to 1 mm if not in config
    num_z_steps = config.get('num_z_steps', 200) # Default to 200 steps if not in config

    # Convert mm values to pixel values for internal calculations
    z_min_pixels = int(z_min_mm / px_mm)
    z_max_pixels = int(z_max_mm / px_mm)
    y_min_pixels = int(y_min_mm / px_mm)
    y_max_pixels = int(y_max_mm / px_mm)

    # Calculate z_step_pixels
    if num_z_steps > 1:
        z_step_pixels = int((z_max_pixels - z_min_pixels) / (num_z_steps - 1))
    else:
        z_step_pixels = 1 # Handle case of single step to avoid division by zero

    # Ensure z_step_pixels is at least 1
    z_step_pixels = max(1, z_step_pixels)


    # Generate or load phase mask
    # assert that either bessel_angle > 0 or args.fresnel_lens_pattern is True or args.mask is provided
    #if int(bessel_angle > 0) + int(bool(initial_phase_mask)) + int(bool(args.mask)) > 1:
    #    raise ValueError("Must specify either a Bessel cone angle, a Fresnel lens pattern, or a phase mask file.")
    if args.mask:
        print(f"Loading phase mask from {args.mask}")
        mask_np = skimage.io.imread(args.mask).astype(np.float32)
        if mask_np.shape != (N, N):
            raise ValueError(f"Loaded mask shape {mask_np.shape} does not match expected {(N, N)}")
    
    elif bessel_angle > 0 and initial_phase_mask == "axicon":
        print(f"Generating axicon phase mask: {N}x{N}, {px_um}um, {wavelength_nm}nm, angle={bessel_angle}deg")
        mask_np = generate_axicon_phase_mask(
            mask_resolution_pixels=(N, N),
            pixel_pitch_um=px_um,
            wavelength_nm=wavelength_nm,
            bessel_cone_angle_degrees=bessel_angle
        )
        
    elif initial_phase_mask == "fresnel_lens":
        print(f"Generating a Fresnel lens phase mask: {N}x{N}, {px_um}um, {wavelength_nm}nm, focal_length={config['focal_length']}m")
        # Generate Fresnel lens phase mask
        yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)
        r = np.sqrt(xx**2 + yy**2) * px_um * 1e-6  # radius in meters
        fresnel_phase = (-np.pi / (wavelength_nm * 1e-9 * config['fresnel_lens_focal_length'])) * (r ** 2)
        mask_np = np.mod(fresnel_phase, 2 * np.pi).astype(np.float32)
   
    elif initial_phase_mask == "empty":
        print("Using empty phase mask (zeros)")
        mask_np = np.zeros((N, N), dtype=np.float32)
        
    else:
        print("No phase mask specified, using empty mask (zeros)")
        mask_np = np.zeros((N, N), dtype=np.float32)

    # save the mask as png figure for easy viewing
    mask_png_path = os.path.join(output_subdir, "mask.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_np, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.title("Phase Mask")
    plt.tight_layout()
    plt.savefig(mask_png_path)
    print(f"Saved phase mask as PNG to {mask_png_path}")

    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)

    config['Nimgs'] = 1
    lens_approach = config['lens_approach']
    phys_layer = PhysicalLayer(config)
    phys_layer.eval()
    with torch.no_grad():
        if lens_approach == 'against_lens':
            print("are you sure you didnt mean fourier lens?")
            output_layer = phys_layer.against_lens(mask_tensor)
        elif lens_approach == 'fourier_lens' or lens_approach == 'convolution':
            output_layer = phys_layer.fourier_lens(mask_tensor)
        elif lens_approach == 'lensless':
            output_layer = phys_layer.lensless(mask_tensor)
        elif lens_approach == '4f':
            output_layer = phys_layer.fourf(mask_tensor)
        else:
            raise ValueError('lens approach not supported')

        # squeeze the output layer to remove singleton dimensions
        output_layer = output_layer.squeeze()

        print("Generating beam profile...")
        output_beam_sections_dir = os.path.join(output_subdir, "beam_sections")
        os.makedirs(output_beam_sections_dir, exist_ok=True)
        beam_profile = phys_layer.generate_beam_cross_section(
            output_layer, output_beam_sections_dir,
            (z_min_pixels, z_max_pixels, z_step_pixels),
            (y_min_pixels, y_max_pixels), asm = asm
        )

        # Save as TIFF (like mask_inference)
        tiff_path = os.path.join(output_subdir, "beam_profile.tiff")
        skimage.io.imsave(tiff_path, (beam_profile))
        print(f"Saved beam profile as TIFF to {tiff_path}")

        # Save as PNG for easy viewing
        png_path = os.path.join(output_subdir, "beam_profile.png")
        plt.figure(figsize=(5, 10))
        plt.imshow(normalize_to_uint16(beam_profile), cmap='hot', aspect='equal')
        plt.colorbar()

        # Set axis labels and ticks in mm
        z_range_mm = np.linspace(z_min_mm, z_max_mm, num_z_steps)
        y_range_mm = np.linspace(y_min_mm, y_max_mm, y_max_pixels - y_min_pixels + 1) 

        plt.title(
            f"Timestamp: {timestamp}\n"
            f"z: {z_min_mm:.2f}mm-{z_max_mm:.2f}mm, steps={num_z_steps},\n"
            f"axicon angle={bessel_angle}°, \n"
            f"lens={lens_approach}, \n"
            f"focal_length_1={1.0e3*config.get('focal_length', 'N/A')}mm, \n"
            f"focal_length_2={1.0e3*config.get('focal_length_2', 'N/A')}mm \n"
            f"fresnel_lens_focal_length={1.0e3*config.get('fresnel_lens_focal_length', 'N/A')}mm\n"
            f"Mask: {initial_phase_mask}\n"
            # display px
            f"Pixel size: {px_um:.2f} um\n"
            f"{'ASM prop' if asm else 'Fresnel prop'}, "
            f"FWHM: {beam_fwhm*10**6:.2f} um"
        )
        plt.xlabel("y (mm)")
        plt.ylabel("z (mm)")
        plt.yticks(
            ticks=np.linspace(0, len(z_range_mm)-1, num=5),
            labels=[f"{z_range_mm[int(i)]:.2f}" for i in np.linspace(0, len(z_range_mm)-1, num=5)]
        )
        # For y-axis ticks, we need to map pixel indices to mm values
        y_tick_pixels = np.linspace(0, len(y_range_mm)-1, num=5)
        y_tick_mm_labels = [f"{y_range_mm[int(j)]:.2f}" for j in y_tick_pixels]
        plt.xticks(
            ticks=y_tick_pixels,
            labels=y_tick_mm_labels
        )
        plt.tight_layout()
        plt.savefig(png_path)
        print(f"Saved beam profile as PNG to {png_path}")

        # save the config used for this test
        config_output_path = os.path.join(output_subdir, "config.yaml")
        with open(config_output_path, "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved configuration to {config_output_path}")

if __name__ == "__main__":
    main()