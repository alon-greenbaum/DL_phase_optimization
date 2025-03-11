import os
import argparse
import pickle
import numpy as np
import skimage.io
import torch
import torch.nn as nn
from datetime import datetime
from data_utils import load_config, makedirs, batch_xyz_to_boolean_grid
from cnn_utils import OpticsDesignCNN
from cnn_utils_unet import OpticsDesignUnet
from physics_utils import PhysicalLayer  # physical model import
from beam_profile_gen import beam_profile_focus, beam_section, phase_mask_gen
from metrics import compute_metrics

# Parse arguments
def main():
    parser = argparse.ArgumentParser(description="Combined inference for physical and CNN mask models.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing config.yaml and labels.pickle")
    parser.add_argument("--epoch", type=int, required=True, help="Desired epoch for the mask (closest available at x*10+1)")
    parser.add_argument("--res_dir", type=str, default="inference_results", help="Directory to save inference outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_inferences", type=int, default=1, help="Number of samples for inference (if 0, use all keys)")
    parser.add_argument("--lens_approach", type=str, default="", help="Override lens_approach in config.yaml for PhysicalLayer")
    parser.add_argument("--empty_mask", action="store_true", help="Run inference with an empty mask")
    parser.add_argument("--paper_mask", type=str, default="", help="File path to paper phase mask for inference")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise in PhysicalLayer inference")
    # Optional CNN model path, will be auto-determined if empty
    parser.add_argument("--model_path", type=str, default="", help="Optional: Path to the CNN pretrained model checkpoint")
    parser.add_argument("--beam_3d_sections", type=str, default="beam_3d_sections", help="Optional: Path to the beam 3d sections file")
    parser.add_argument("--generate_beam_profile", action="store_true", help="Generate beam profile for the input mask (default: off)")
    parser.add_argument("--img_at_end_epoch", action="store_true", help="Use the mask image at the end of the epoch (default: off)")
    parser.add_argument("--max_intensity", type=float, default=5.0e+4, help="Maximum intensity for the mask (default: 1.0)")
    args = parser.parse_args()

    # Automatically determine CNN model path if not provided
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
        print(f"Automatically using CNN model: {args.model_path}")

    # Load configuration and set device
    config_path = os.path.join(args.input_dir, "config.yaml")
    config = load_config(config_path)
    config['device'] = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config['inference_epoch'] = args.epoch
    if args.max_intensity:
        config['max_intensity'] = args.max_intensity
    if args.lens_approach:
        config['lens_approach'] = args.lens_approach
    if args.no_noise:
        config['skip_noise'] = True
        
    if type(config['px']) == str:
        config['px'] = float(config['px'])

    # Load mask from tiff file (for both models)
    if args.img_at_end_epoch:
        mask_filename = f"mask_phase_epoch_{args.epoch-1}_249.tiff"
    else:
        mask_filename = f"mask_phase_epoch_{args.epoch-1}_0.tiff"
        
    mask_path = os.path.join(args.input_dir, mask_filename)
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(config['device'])
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)

    # Pre-load paper mask if provided
    if args.paper_mask:
        paper_mask_np = skimage.io.imread(args.paper_mask)
        paper_mask_tensor = torch.from_numpy(paper_mask_np).type(torch.FloatTensor).to(config['device'])
        paper_mask_param = torch.nn.Parameter(paper_mask_tensor, requires_grad=False)

    # Load labels
    labels_path = os.path.join(args.input_dir, "labels.pickle")
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    all_keys = sorted(labels_dict.keys(), key=lambda x: int(x))
    if args.num_inferences > 0 and args.num_inferences < len(all_keys):
        indices = np.linspace(0, len(all_keys)-1, args.num_inferences).astype(int)
        selected_keys = [all_keys[i] for i in indices]
    else:
        selected_keys = all_keys

    # Instantiate PhysicalLayer model (no checkpoint loading assumed)
    phys_model = PhysicalLayer(config).to(config['device'])
    phys_model.eval()

    # Instantiate CNN model based on config and load checkpoint
    if config.get("use_unet", False):
        cnn_model = OpticsDesignUnet(config)
        print("Instantiated OpticsDesignUnet")
    else:
        cnn_model = OpticsDesignCNN(config)
        print("Instantiated OpticsDesignCNN")
    cnn_model.to(config['device'])
    cnn_model.load_state_dict(torch.load(args.model_path, map_location=config['device']))
    cnn_model.eval()

    # Create output directory for inference results using current datetime.
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, dt_str)
    makedirs(out_dir)

    # Save updated configuration in out_dir
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # --- New: Optionally generate and save beam profile for the input mask ---
    if args.generate_beam_profile:
        input_mask_path = os.path.join(out_dir, "input_mask.tiff")
        skimage.io.imsave(input_mask_path, mask_np)
        print(f"Saved input mask to {input_mask_path}")
    
        beam_3d_sections_filepath = os.path.join(out_dir, args.beam_3d_sections)
        if not os.path.exists(beam_3d_sections_filepath):
            os.makedirs(beam_3d_sections_filepath)
        mask_np_for_beam = mask_tensor.cpu().numpy()
        mask_real = phase_mask_gen()
        mask_param_for_beam = mask_real*np.exp(1j*mask_np_for_beam)
        beam_focused = beam_profile_focus(mask_param_for_beam)
        beam_profile = beam_section(beam_focused, beam_3d_sections_filepath)
        beam_profile_out_path = os.path.join(out_dir, "beam_profile.tiff")
        skimage.io.imsave(beam_profile_out_path, (beam_profile/1e6).astype(np.uint16))
        print(f"Saved beam profile for input mask to {beam_profile_out_path}")
    # --- End new code ---

    # Run inference for each selected key
    for key in selected_keys:
        data = labels_dict[key]
        xyz_np = data['xyz']
        xyz = torch.tensor(xyz_np, dtype=torch.float32).to(config['device'])
        Nphotons = torch.tensor(data['N'], dtype=torch.float32).to(config['device'])

        with torch.no_grad():
            # Physical layer inference
            out_phys = phys_model(mask_param, xyz, Nphotons)
            # CNN layer inference
            out_cnn = cnn_model(mask_param, xyz, Nphotons)
        
        # Save physical inference output
        phys_img = out_phys.detach().cpu().squeeze().numpy()
        phys_path = os.path.join(out_dir, f"inference_phys_{key}.tiff")
        skimage.io.imsave(phys_path, phys_img[0,:,:])
        print(f"Saved physical inference for key {key} to {phys_path}")

        # Save CNN inference output
        cnn_img = out_cnn.detach().cpu().squeeze().numpy()
        cnn_path = os.path.join(out_dir, f"inference_cnn_{key}.tiff")
        skimage.io.imsave(cnn_path, cnn_img[0,:,:])
        print(f"Saved CNN inference for key {key} to {cnn_path}")

        # Save ground truth
        gt_img = batch_xyz_to_boolean_grid(xyz_np, config)
        if torch.is_tensor(gt_img):
            gt_img = gt_img.detach().cpu().numpy()
        if gt_img.dtype == np.bool_:
            gt_img = (gt_img.astype(np.uint8)) * 255
        gt_path = os.path.join(out_dir, f"ground_truth_{key}.tiff")
        skimage.io.imsave(gt_path, gt_img[0,:,:])
        print(f"Saved ground truth for key {key} to {gt_path}")

        # --- New: Compute and log performance metrics ---
        gt_full = gt_img[0, :, :]
        cnn_full = cnn_img[0, :, :]
        precision, recall, f1 = compute_metrics(gt_full, cnn_full)
        print(f"Full image: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        """
        H = gt_full.shape[0]
        top_gt = gt_full[0,:H//3, :]
        top_pred = cnn_full[0,:H//3, :]
        precision_top, recall_top, f1_top = compute_metrics(top_gt, top_pred)
        print(f"Top third:    Precision: {precision_top:.4f}, Recall: {recall_top:.4f}, F1: {f1_top:.4f}")
        
        middle_gt = gt_full[H//3:2*H//3, :]
        middle_pred = cnn_full[H//3:2*H//3, :]
        precision_mid, recall_mid, f1_mid = compute_metrics(middle_gt, middle_pred)
        print(f"Middle third: Precision: {precision_mid:.4f}, Recall: {recall_mid:.4f}, F1: {f1_mid:.4f}")
        
        bottom_gt = gt_full[2*H//3:, :]
        bottom_pred = cnn_full[2*H//3:, :]  # Updated slice to match expected dimensions
        precision_bot, recall_bot, f1_bot = compute_metrics(bottom_gt, bottom_pred)
        print(f"Bottom third: Precision: {precision_bot:.4f}, Recall: {recall_bot:.4f}, F1: {f1_bot:.4f}")
        # --- End new code ---
        """
        # Optional: Empty mask inference
        if args.empty_mask:
            empty_mask_tensor = torch.zeros_like(mask_tensor)
            empty_mask_param = torch.nn.Parameter(empty_mask_tensor, requires_grad=False)
            with torch.no_grad():
                out_empty_phys = phys_model(empty_mask_param, xyz, Nphotons)
                out_empty_cnn = cnn_model(empty_mask_param, xyz, Nphotons)
            empty_phys_img = out_empty_phys.detach().cpu().squeeze().numpy()
            empty_cnn_img = out_empty_cnn.detach().cpu().squeeze().numpy()
            empty_phys_path = os.path.join(out_dir, f"inference_empty_phys_{key}.tiff")
            empty_cnn_path = os.path.join(out_dir, f"inference_empty_cnn_{key}.tiff")
            skimage.io.imsave(empty_phys_path, empty_phys_img[0,:,:])
            skimage.io.imsave(empty_cnn_path, empty_cnn_img[0,:,:])
            print(f"Saved empty mask inference for key {key} to {empty_phys_path} and {empty_cnn_path}")

        # Optional: Paper mask inference if provided
        if args.paper_mask:
            with torch.no_grad():
                out_paper_phys = phys_model(paper_mask_param, xyz, Nphotons)
                out_paper_cnn = cnn_model(paper_mask_param, xyz, Nphotons)
            paper_phys_img = out_paper_phys.detach().cpu().squeeze().numpy()
            paper_cnn_img = out_paper_cnn.detach().cpu().squeeze().numpy()
            paper_phys_path = os.path.join(out_dir, f"inference_paper_phys_{key}.tiff")
            paper_cnn_path = os.path.join(out_dir, f"inference_paper_cnn_{key}.tiff")
            skimage.io.imsave(paper_phys_path, paper_phys_img[0,:,:])
            skimage.io.imsave(paper_cnn_path, paper_cnn_img[0,:,:])
            print(f"Saved paper mask inference for key {key} to {paper_phys_path} and {paper_cnn_path}")

if __name__ == "__main__":
    main()
