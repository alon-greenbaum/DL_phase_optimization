import os
import argparse
import numpy as np
import skimage.io
from datetime import datetime
from beam_profile_gen import beam_profile_focus, beam_section, phase_mask_gen

def main():
    parser = argparse.ArgumentParser(description="Generate beam profile from input mask.")
    parser.add_argument("--input_dir", type=str, required=False, help="Directory containing the input mask tiff")
    parser.add_argument("--epoch", type=int, required=False, help="Epoch for selecting the mask file")
    parser.add_argument("--res_dir", type=str, default="beam_profile_results", help="Directory to save beam profile outputs")
    parser.add_argument("--beam_3d_sections", type=str, default="beam_3d_sections", help="Subfolder for beam section slices")
    # New optional argument to define mask filepath directly:
    parser.add_argument("--mask_filepath", type=str, default="", help="Optional: Full filepath to input mask. Overrides input_dir and epoch if provided.")
    # New optional argument for dummy mask generation
    parser.add_argument("--dummy_mask", action="store_true", help="Use a 500x500 mask of zeros instead of reading an input mask")
    # New optional arguments for z_min and z_max:
    parser.add_argument("--z_min", type=int, default=-201, help="Minimum z value for beam section generation")
    parser.add_argument("--z_max", type=int, default=201, help="Maximum z value for beam section generation")
    args = parser.parse_args()
    
    # Determine mask path or create dummy mask:
    if args.dummy_mask:
        print("Using dummy 500x500 mask of zeros.")
        mask_np = np.zeros((500,500), dtype=np.float32)
    else:
        if args.mask_filepath:
            mask_path = args.mask_filepath
            print(f"Using user-provided mask file: {mask_path}")
        else:
            mask_filename = f"mask_phase_epoch_{args.epoch-1}_0.tiff"
            mask_path = os.path.join(args.input_dir, mask_filename)
            print(f"Automatically using mask file: {mask_path}")
        mask_np = skimage.io.imread(mask_path)
    
    # Generate complex mask: combine generated phase mask with loaded mask
    mask_real = phase_mask_gen()
    mask_complex = mask_real * np.exp(1j * mask_np)
    
    # Compute beam focus and beam sections
    beam_focused = beam_profile_focus(mask_complex)
    
    # Create output directory and subfolder for beam sections
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, dt_str)
    os.makedirs(out_dir, exist_ok=True)
    beam_sections_dir = os.path.join(out_dir, args.beam_3d_sections)
    os.makedirs(beam_sections_dir, exist_ok=True)
    
    # --- New: Save configuration including z_min and z_max ---
    config = {
        "input_dir": args.input_dir,
        "epoch": args.epoch,
        "mask_filepath": args.mask_filepath,
        "dummy_mask": args.dummy_mask,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "beam_3d_sections": args.beam_3d_sections
    }
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    # --- End new code ---
    
    # --- New: Save the mask used to create the beam profile ---
    mask_used_path = os.path.join(out_dir, "input_mask.tiff")
    skimage.io.imsave(mask_used_path, mask_np)
    print(f"Saved input mask used to {mask_used_path}")
    # --- End new code ---
    
    # Compute beam section profiles using provided z_min and z_max:
    beam_profile = beam_section(beam_focused, beam_sections_dir, args.z_min, args.z_max)
    
    # Save overall beam profile as a TIFF file
    beam_profile_out_path = os.path.join(out_dir, "beam_profile.tiff")
    # Transpose and adjust scaling; save as uint16 for visualization
    skimage.io.imsave(beam_profile_out_path, (beam_profile.T/1e6).astype(np.uint16))
    print(f"Saved beam profile to {beam_profile_out_path}")
    print(f"Beam section slices saved in {beam_sections_dir}")

if __name__ == "__main__":
    main()
