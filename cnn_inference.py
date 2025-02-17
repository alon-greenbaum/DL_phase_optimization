# -*- coding: utf-8 -*-
import numpy as np
import skimage.io
import os
import torch
import torch.nn as nn
import pickle
from psf_gen import apply_blur_kernel
from data_utils import PhasesOnlineDataset, savePhaseMask
from cnn_utils import OpticsDesignCNN
from loss_utils import KDE_loss3D, jaccard_coeff
from beam_profile_gen import phase_gen, phase_mask_gen
import argparse
from datetime import datetime
from data_utils import load_config, makedirs, batch_xyz_to_boolean_grid
from cnn_utils_unet import OpticsDesignUnet

N = 500 # grid size
px = 1e-6 # pixel size (um)
focal_length = 6e-3
wavelength = 0.561e-6
refractive_index = 1.0
psf_width_pixels = 101
pixel_size_meters = 1e-6
psf_width_meters = psf_width_pixels * pixel_size_meters
numerical_aperture = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Inference script for OpticsDesign model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing config.yaml and labels.pickle")
    # Removed required model_path; it will be computed automatically.
    parser.add_argument("--model_path", type=str, default="", help="Optional: Path to the pretrained model checkpoint")
    parser.add_argument("--epoch", type=int, required=True, help="Desired epoch for the model (closest available at x*10+1)")
    parser.add_argument("--res_dir", type=str, default="results", help="Directory to save inference outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_inferences", type=int, default=1, help="Number of samples for inference (if 0, use all keys)")
    args = parser.parse_args()
    
    # Automatically determine model path if not provided
    if not args.model_path:
        # x is an integer that begins at 0 and increases by 1.
        x0 = int((args.epoch - 1) / 10)
        candidate_low = x0 * 10 + 1
        candidate_high = (x0 + 1) * 10 + 1
        if abs(args.epoch - candidate_low) <= abs(candidate_high - args.epoch):
            chosen_epoch = candidate_low
        else:
            chosen_epoch = candidate_high
        model_file = f"net_{chosen_epoch - 1}.pt"
        args.model_path = os.path.join(args.input_dir, model_file)
        print(f"Automatically using model: {args.model_path}")
    
    # Load configuration and set device
    config_path = os.path.join(args.input_dir, "config.yaml")
    config = load_config(config_path)
    config['device'] = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config['inference_epoch'] = args.epoch

    # Load mask from tiff file (similar to physics_inference.py)
    mask_filename = f"mask_phase_epoch_{args.epoch-1}_0.tiff"
    mask_path = os.path.join(args.input_dir, mask_filename)
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(config['device'])
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)
    
    # Load labels
    labels_path = os.path.join(args.input_dir, "labels.pickle")
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    
    all_keys = sorted(labels_dict.keys(), key=lambda x: int(x))  # assuming keys are numeric strings
    if args.num_inferences > 0 and args.num_inferences < len(all_keys):
        indices = np.linspace(0, len(all_keys) - 1, args.num_inferences).astype(int)
        selected_keys = [all_keys[i] for i in indices]
    else:
        selected_keys = all_keys
    
    # Instantiate model based on configuration
    if config.get("use_unet", False):
        model = OpticsDesignUnet(config)
        print("Instantiated OpticsDesignUnet")
    else:
        model = OpticsDesignCNN(config)
        print("Instantiated OpticsDesignCNN")
    model.to(config['device'])
    
    # Load the pretrained model checkpoint
    model.load_state_dict(torch.load(args.model_path, map_location=config['device']))
    model.eval()
    
    # Create output directory for inference results using current datetime.
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, dt_str)
    makedirs(out_dir)
    
    # Save updated configuration in out_dir, including inference epoch.
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Run inference and save outputs
    for key in selected_keys:
        data = labels_dict[key]
        xyz_np = data['xyz']
        # Convert xyz to tensor on proper device
        xyz = torch.tensor(xyz_np, dtype=torch.float32).to(config['device'])
        Nphotons = torch.tensor(data['N'], dtype=torch.float32).to(config['device'])
        
        with torch.no_grad():
            # Updated: pass mask_param as the first argument to model
            outputs = model(mask_param, xyz, Nphotons)
        out_img = outputs.detach().cpu().squeeze().numpy()
        out_path = os.path.join(out_dir, f"inference_{key}.tiff")
        skimage.io.imsave(out_path, out_img)
        print(f"Saved inference for key {key} to {out_path}")
        
        # Save ground truth label from boolean grid
        gt_img = batch_xyz_to_boolean_grid(xyz_np, config)
        if torch.is_tensor(gt_img):
            gt_img = gt_img.detach().cpu().numpy()
        if gt_img.dtype == np.bool_:
            gt_img = (gt_img.astype(np.uint8)) * 255
        gt_path = os.path.join(out_dir, f"ground_truth_{key}.tiff")
        skimage.io.imsave(gt_path, gt_img)
        print(f"Saved ground truth for key {key} to {gt_path}")

if __name__ == "__main__":
    main()




