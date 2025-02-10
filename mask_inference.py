import os
import argparse
import pickle
import torch
import numpy as np
import skimage.io
from datetime import datetime
from data_utils import load_config, makedirs, batch_xyz_to_boolean_grid
from physics_utils import PhysicalLayer

def main():
    parser = argparse.ArgumentParser(description="Inference script for PhysicalLayer.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing config.yaml and labels.pickle")
    parser.add_argument("--epoch", type=int, required=True, help="Desired epoch for the mask (e.g. 10)")
    parser.add_argument("--res_dir", type=str, default="results", help="Directory to save inference outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_inferences", type=int, default=10, help="Number of inferences to perform. If 0, use all keys.")
    args = parser.parse_args()

    # Load configuration and set device
    config_path = os.path.join(args.input_dir, "config.yaml")
    config = load_config(config_path)
    config['device'] = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build mask file path automatically using the epoch input
    mask_filename = f"mask_phase_epoch_{args.epoch-1}_0.tiff"
    mask_path = os.path.join(args.input_dir, mask_filename)
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(config['device'])
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)

    # Load labels
    labels_path = os.path.join(args.input_dir, "labels.pickle")
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)

    # Select keys evenly spaced out if desired number is provided
    all_keys = sorted(labels_dict.keys(), key=lambda x: int(x))  # sorted assuming string keys represent ints
    if args.num_inferences > 0 and args.num_inferences < len(all_keys):
        indices = np.linspace(0, len(all_keys) - 1, args.num_inferences).astype(int)
        selected_keys = [all_keys[i] for i in indices]
    else:
        selected_keys = all_keys

    # Instantiate the PhysicalLayer model in evaluation mode
    phys_layer = PhysicalLayer(config).to(config['device'])
    phys_layer.eval()

    # Create output directory for inference results: subdir in res_dir with the same name as the last part of input_dir
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    out_dir = os.path.join(args.res_dir, input_dir_name)
    makedirs(out_dir)

    # run inference and save the output image and corresponding ground truth for selected keys
    for key in selected_keys:
        data = labels_dict[key]
        # retain original xyz numpy array for ground truth
        xyz_np = data['xyz']
        xyz = torch.tensor(xyz_np, dtype=torch.float32).to(config['device'])
        Nphotons = torch.tensor(data['N'], dtype=torch.float32).to(config['device'])
       
        with torch.no_grad():
            output = phys_layer(mask_param, xyz, Nphotons)  # placeholder for unused Nphotons
        # Assume output shape is (1, 1, H, W); squeeze extra dimensions
        out_img = output.detach().cpu().squeeze().numpy()
        
        # Optionally normalize to 0-255 for visualization
        # out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"inference_{key}.tiff")
        skimage.io.imsave(out_path, out_img)
        print(f"Saved image for key {key} to {out_path}")
        
        # Compute and save the ground truth image using batch_xyz_to_boolean_grid
        gt_img = batch_xyz_to_boolean_grid(xyz_np, config)
        # Convert tensor output to numpy array if needed
        if torch.is_tensor(gt_img):
            gt_img = gt_img.detach().cpu().numpy()
            
        # Convert boolean image to uint8 for saving (if necessary)
        if gt_img.dtype == np.bool_:
            gt_img = (gt_img.astype(np.uint8)) * 255
            
        gt_path = os.path.join(out_dir, f"ground_truth_{key}.tiff")
        skimage.io.imsave(gt_path, gt_img)
        print(f"Saved ground truth for key {key} to {gt_path}")

if __name__ == "__main__":
    main()
