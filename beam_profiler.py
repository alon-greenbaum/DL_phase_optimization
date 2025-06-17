import os
import argparse
import numpy as np
import torch
import skimage.io
import matplotlib.pyplot as plt
from datetime import datetime
from data_utils import load_config
#from beam_profile_gen import BeamProfiler
from bessel import generate_axicon_phase_mask
from physics_utils import PhysicalLayer

def main():
    parser = argparse.ArgumentParser(description="Test light propagation with optional axicon phase mask and save beam profile.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--mask", type=str, default="", help="Optional: path to phase mask tiff (default: zeros)")
    #parser.add_argument("--axicon", action="store_true", help="Use axicon phase mask (Bessel beam)")
    parser.add_argument("--output_dir", type=str, default="beam_profile_test", help="Output directory")
    parser.add_argument("--z_min", type=int, default=-10000)
    parser.add_argument("--z_max", type=int, default=10000)
    parser.add_argument("--z_step", type=int, default=50)
    parser.add_argument("--y_min", type=int, default=-100)
    parser.add_argument("--y_max", type=int, default=100)
    parser.add_argument("--fresnel_lens_pattern", action="store_true", help="Use Fresnel lens phase mask")
    args = parser.parse_args()

    # Create a timestamped subfolder output dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_subdir, exist_ok=True)

    config = load_config(args.config)
    N = config['N']
    px = config['px']  # pixel size in meters
    px_mm = config['px'] * 1e3 if config['px'] < 1e-3 else config['px']  # px in mm
    px_um = config['px'] * 1e6 if config['px'] < 1e-3 else config['px']  # px in um
    wavelength_nm = config['wavelength'] * 1e9 if config['wavelength'] < 1e-6 else config['wavelength']  # wavelength in nm
    beam_fwhm = config['laser_beam_FWHC']
    bessel_angle = config['bessel_cone_angle_degrees']
    config['device'] = 'cpu' if torch.cuda.is_available() else 'cpu'
    device = config['device']
    asm = config.get('angular_spectrum_method', True)

    # Generate or load phase mask
    if bessel_angle > 0:
        print(f"Generating axicon phase mask: {N}x{N}, {px_um}um, {wavelength_nm}nm, angle={bessel_angle}deg")
        mask_np = generate_axicon_phase_mask(
            mask_resolution_pixels=(N, N),
            pixel_pitch_um=px_um,
            wavelength_nm=wavelength_nm,
            bessel_cone_angle_degrees=bessel_angle
        )
    elif args.fresnel_lens_pattern:
        print(f"Generating a Fresnel lens phase mask: {N}x{N}, {px_um}um, {wavelength_nm}nm, focal_length={config['focal_length']}m")
        # Generate Fresnel lens phase mask
        yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)
        r = np.sqrt(xx**2 + yy**2) * px_um * 1e-6  # radius in meters
        fresnel_phase = (-np.pi / (wavelength_nm * 1e-9 * config['focal_length'])) * (r ** 2)
        mask_np = np.mod(fresnel_phase, 2 * np.pi).astype(np.float32)
    elif args.mask:
        print(f"Loading phase mask from {args.mask}")
        mask_np = skimage.io.imread(args.mask).astype(np.float32)
        if mask_np.shape != (N, N):
            raise ValueError(f"Loaded mask shape {mask_np.shape} does not match expected {(N, N)}")
    else:
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
        
        # squeeze the output layer to remove singleton dimensions
        output_layer = output_layer.squeeze() 
        #output_layer = output_layer.detach().cpu().numpy()
        #output_layer = np.squeeze(output_layer)

        print("Generating beam profile...") 
        output_beam_sections_dir = os.path.join(output_subdir, "beam_sections")
        os.makedirs(output_beam_sections_dir, exist_ok=True)
        beam_profile = phys_layer.generate_beam_cross_section(
            output_layer, output_beam_sections_dir, (args.z_min, args.z_max, args.z_step), (args.y_min, args.y_max), asm = asm
        )

        # Save as TIFF (like mask_inference)
        tiff_path = os.path.join(output_subdir, "beam_profile.tiff")
        skimage.io.imsave(tiff_path, (beam_profile))
        print(f"Saved beam profile as TIFF to {tiff_path}")

        # Save as PNG for easy viewing
        png_path = os.path.join(output_subdir, "beam_profile.png")
        plt.figure(figsize=(5, 10))
        plt.imshow(phys_layer.normalize_to_uint16(beam_profile), cmap='hot', aspect='equal')
        plt.colorbar()
        # Set axis labels and ticks in mm
        z_range = np.arange(args.z_min* px_mm, args.z_max* px_mm, args.z_step * px_mm)  # mm
        y_range = np.arange(args.y_min*px_mm, px_mm * (args.y_max), px_mm)   # mm

        plt.title(
            f"Beam Profile\n"
            f"z: {args.z_min*px_mm}mm-{args.z_max*px_mm}mm, step={args.z_step*px_mm}mm,\n"
            f"axicon angle={bessel_angle}°, \n"
            f"lens={lens_approach}, \n"
            f"focal_length_1={1000*config.get('focal_length', 'N/A')}mm, \n"
            f"focal_length_2={1000*config.get('focal_length_2', 'N/A')}mm \n"
            f"{'Fresnel lens' if args.fresnel_lens_pattern else ''}"
            f"{'ASM prop' if asm else 'Fresnel prop'}"
        )
        plt.xlabel("y (mm)")
        plt.ylabel("z (mm)")
        plt.yticks(
            ticks=np.linspace(0, len(z_range)-1, num=5),
            labels=[f"{z_range[int(i)]:.2f}" for i in np.linspace(0, len(z_range)-1, num=5)]
        )
        plt.xticks(
            ticks=np.linspace(0, len(y_range)-1, num=5),
            labels=[f"{y_range[int(i)]:.2f}" for i in np.linspace(0, len(y_range)-1, num=5)]
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