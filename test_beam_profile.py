import os
import argparse
import numpy as np
import torch
import skimage.io
import matplotlib.pyplot as plt
from datetime import datetime
from data_utils import load_config
from beam_profile_gen import beam_profile_focus, beam_section, phase_mask_gen
from bessel import generate_axicon_phase_mask
from physics_utils import PhysicalLayer

def main():
    parser = argparse.ArgumentParser(description="Test light propagation with optional axicon phase mask and save beam profile.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--mask", type=str, default="", help="Optional: path to phase mask tiff (default: zeros)")
    parser.add_argument("--axicon", action="store_true", help="Use axicon phase mask (Bessel beam)")
    parser.add_argument("--output_dir", type=str, default="beam_profile_test", help="Output directory")
    parser.add_argument("--x_min", type=int, default=-100)
    parser.add_argument("--x_max", type=int, default=100)
    parser.add_argument("--y_min", type=int, default=-50)
    parser.add_argument("--y_max", type=int, default=50)
    args = parser.parse_args()

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_subdir, exist_ok=True)

    config = load_config(args.config)
    N = config['N']
    px_um = config['px'] * 1e6 if config['px'] < 1e-3 else config['px']  # px in um
    wavelength_nm = config['wavelength'] * 1e9 if config['wavelength'] < 1e-6 else config['wavelength']  # wavelength in nm
    beam_fwhm = config['laser_beam_FWHC']
    bessel_angle = config['bessel_cone_angle_degrees']
    config['device'] = 'cpu' 
    device = config['device']

    # Generate or load phase mask
    if args.axicon:
        print(f"Generating axicon phase mask: {N}x{N}, {px_um}um, {wavelength_nm}nm, angle={bessel_angle}deg")
        mask_np = generate_axicon_phase_mask(
            mask_resolution_pixels=(N, N),
            pixel_pitch_um=px_um,
            wavelength_nm=wavelength_nm,
            bessel_cone_angle_degrees=bessel_angle
        )
    elif args.mask and os.path.exists(args.mask):
        mask_np = skimage.io.imread(args.mask)
    else:
        mask_np = np.zeros((N, N), dtype=np.float32)

    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)

    config['device'] = 'cpu'  # Use CPU for testing
    config['Nimgs'] = 1
    lens_approach = config['lens_approach']
    phys_layer = PhysicalLayer(config)
    phys_layer.eval()
    if lens_approach == 'fourier' or lens_approach == 'against_lens':
        output_layer = phys_layer.a(mask_tensor)
    elif lens_approach == 'fourier_lens' or lens_approach == 'convolution':
        output_layer = phys_layer.fourier_lens(mask_tensor)
    elif lens_approach == 'lensless':
        output_layer = phys_layer.lensless(mask_tensor)
    elif lens_approach == '4f':
        output_layer = phys_layer.fourf(mask_tensor)
    else:
        raise ValueError('lens approach not supported')
    
    output_layer = output_layer.detach().cpu().numpy()
    output_layer = np.squeeze(output_layer)

    print("Generating beam profile...") 
    output_beam_sections_dir = os.path.join(output_subdir, "beam_sections")
    os.makedirs(output_beam_sections_dir, exist_ok=True)
    beam_profile = beam_section(
        output_layer, output_beam_sections_dir, args.x_min, args.x_max, args.y_min, args.y_max
    )

    # Save as TIFF (like mask_inference)
    tiff_path = os.path.join(output_subdir, "beam_profile.tiff")
    skimage.io.imsave(tiff_path, (beam_profile).astype(np.uint16))
    print(f"Saved beam profile as TIFF to {tiff_path}")

    # Save as PNG for easy viewing
    png_path = os.path.join(output_subdir, "beam_profile.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(beam_profile, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title("Beam Profile")
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