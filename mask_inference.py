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
    parser = argparse.ArgumentParser(description="Inference script for the PhysicalLayer.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing config.yaml and labels.pickle")
    parser.add_argument("--epoch", type=int, required=True, help="Desired epoch for the mask (e.g. 10)")
    parser.add_argument("--res_dir", type=str, default="results", help="Directory to save inference outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_inferences", type=int, default=1, help="Number of inferences to perform. If 0, use all keys.")
    parser.add_argument("--lens_approach", type=str, default="", help="Override lens_approach in config.yaml")
    parser.add_argument("--empty_mask", action="store_true", help="Generate images for an empty phase mask")
    parser.add_argument("--paper_mask", type=str, default="", help="File path to paper phase mask for inference")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise addition in PhysicalLayer inference")
    args = parser.parse_args()

    # Load configuration and set device
    config_path = os.path.join(args.input_dir, "config.yaml")
    config = load_config(config_path)
    
    config['training_lens_approach'] = config['lens_approach']
    config['device'] = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.lens_approach:
        config['lens_approach'] = args.lens_approach  # override lens_approach if provided

    # load seed from training config and set random seed
    random_seed = config['random_seed']
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # update config with new argument value
    config['skip_noise'] = args.no_noise

    # Build mask file path automatically using the epoch input
    mask_filename = f"mask_phase_epoch_{args.epoch-1}_0.tiff"
    mask_path = os.path.join(args.input_dir, mask_filename)
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(config['device'])
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)

    # Pre-load the paper phase mask if provided
    if args.paper_mask:
        paper_mask_np = skimage.io.imread(args.paper_mask)
        paper_mask_tensor = torch.from_numpy(paper_mask_np).type(torch.FloatTensor).to(config['device'])
        paper_mask_param = torch.nn.Parameter(paper_mask_tensor, requires_grad=False)

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

    # Create output directory for inference results: use res_dir and current datetime
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, dt_str)
    makedirs(out_dir)
    
    # Add command-line arguments to config and save updated config to config.yaml in out_dir
    config['lens_approach'] = config.get('lens_approach', "")
    config['num_inferences'] = args.num_inferences
    config['inference_epoch'] = args.epoch
    config['inference_input_dir'] = args.input_dir
    config['paper_mask'] = args.paper_mask
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

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
        
        # Generate images for an empty phase mask if enabled
        if args.empty_mask:
            empty_mask_tensor = torch.zeros_like(mask_tensor)
            empty_mask_param = torch.nn.Parameter(empty_mask_tensor, requires_grad=False)
            with torch.no_grad():
                empty_output = phys_layer(empty_mask_param, xyz, Nphotons)
            empty_out_img = empty_output.detach().cpu().squeeze().numpy()
            empty_out_path = os.path.join(out_dir, f"inference_empty_{key}.tiff")
            skimage.io.imsave(empty_out_path, empty_out_img)
            print(f"Saved empty phase inference image for key {key} to {empty_out_path}")
            
        # Generate inference using the paper phase mask if provided
        if args.paper_mask:
            with torch.no_grad():
                paper_output = phys_layer(paper_mask_param, xyz, Nphotons)
            paper_out_img = paper_output.detach().cpu().squeeze().numpy()
            paper_out_path = os.path.join(out_dir, f"inference_paper_{key}.tiff")
            skimage.io.imsave(paper_out_path, paper_out_img)
            print(f"Saved paper phase inference image for key {key} to {paper_out_path}")

if __name__ == "__main__":
    main()
