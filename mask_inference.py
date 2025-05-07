import os
import argparse
import pickle
import numpy as np
import skimage.io
import torch
from datetime import datetime
from data_utils import load_config, makedirs, batch_xyz_to_boolean_grid, img_save_tiff, find_image_with_wildcard, batch_xyz_to_3_class_grid, save_3d_volume_as_tiffs, batch_xyz_to_3class_volume
from cnn_utils import OpticsDesignCNN
from cnn_utils_unet import OpticsDesignUnet
from physics_utils import PhysicalLayer  # physical model import
from beam_profile_gen import beam_profile_focus, beam_section, phase_mask_gen
from metrics import compute_and_log_metrics, save_heatmap
import matplotlib.pyplot as plt

def run_inference(model, mask_param, xyz, model_type = None):
    """
    Runs inference using the given model.

    Args:
        model (torch.nn.Module): The model to use for inference.
        mask_param (torch.nn.Parameter): The phase mask parameter.
        xyz (torch.Tensor): The XYZ coordinates.
        Nphotons (torch.Tensor): The number of photons.
        device (torch.device): The device to run inference on.

    Returns:
        torch.Tensor: The output of the model.
    """
    with torch.no_grad():
        out = model(mask_param, xyz)
    if model_type == "cnn":
        out = torch.sigmoid(out)
    if model_type == "cnn_3_class":
        out = torch.softmax(out, dim=1)
        out = torch.argmax(out,dim=1)
    return out.detach().cpu().squeeze().numpy()

def main():
    """
    Main function to run combined inference for physical and CNN mask models.
    """
    # Parse arguments
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
    parser.add_argument("--model_path", type=str, default="", help="Optional: Path to the CNN pretrained model checkpoint")
    parser.add_argument("--beam_3d_sections", type=str, default="beam_3d_sections", help="Optional: Path to the beam 3d sections file")
    parser.add_argument("--generate_beam_profile", action="store_true", help="Generate beam profile for the input mask (default: off)")
    parser.add_argument("--x_min", type=int, default=-100, help="Minimum z value for beam section generation")
    parser.add_argument("--x_max", type=int, default=100, help="Maximum z value for beam section generation")
    parser.add_argument("--y_min", type=int, default=-50, help="Minimum y value for beam section generation")
    parser.add_argument("--y_max", type=int, default=50, help="Maximum y value for beam section generation")
    parser.add_argument("--max_intensity", type=float, help="Maximum intensity for the mask")
    parser.add_argument("--bead_volume", action="store_true", help="Save bead volume as tiff files")
    parser.add_argument("--plot_train_loss", action="store_true", help="Plot the training loss over time from train_losses.txt in input_dir")
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
        
    
    display_batch = 0  # Display the first image in the batch
    
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
        
    config["model_path"] = args.model_path

    # Load mask from tiff file (for both models)
    mask_path = find_image_with_wildcard(args.input_dir, f"mask_phase_epoch_{args.epoch-1}_", "tiff")
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
    cnn_config = config.copy()
    cnn_config['skip_noise'] = False
    if config.get("use_unet", False):
        cnn_model = OpticsDesignUnet(cnn_config)
        print("Instantiated OpticsDesignUnet")
    else:
        cnn_model = OpticsDesignCNN(cnn_config)
        print("Instantiated OpticsDesignCNN")
    cnn_model.to(config['device'])
    cnn_model.load_state_dict(torch.load(args.model_path, map_location=config['device']))
    cnn_model.eval()

    # Create output directory for inference results using current datetime.
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, dt_str)
    makedirs(out_dir)
    
    if args.plot_train_loss:
        loss_file = os.path.join(args.input_dir, "train_losses.txt")
        if not os.path.exists(loss_file):
            print(f"train_losses.txt not found in {args.input_dir}")
        else:
            with open(loss_file, "r") as f:
                losses = [float(line.strip()) for line in f if line.strip()]
            plt.figure()
            plt.plot(losses, label="Training Loss")
            plt.xlabel("Epoch or Iteration")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Time")
            plt.legend()
            save_path = os.path.join(out_dir, "train_loss.png")
            plt.savefig(save_path)
            print(f"Training loss plot saved to {save_path}")

    # Save updated configuration in out_dir
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for batch, value in config.items():
            f.write(f"{batch}: {value}\n")
    
    input_mask_path = os.path.join(out_dir, "input_mask.tiff")
    skimage.io.imsave(input_mask_path, mask_np)
    print(f"Saved input mask to {input_mask_path}")
        
    # Generate and save beam profile for the input mask
    if args.generate_beam_profile:
        beam_3d_sections_filepath = os.path.join(out_dir, args.beam_3d_sections)
        if not os.path.exists(beam_3d_sections_filepath):
            os.makedirs(beam_3d_sections_filepath)
        mask_np_for_beam = mask_tensor.cpu().numpy()
        mask_real = phase_mask_gen()
        mask_param_for_beam = mask_real*np.exp(1j*mask_np_for_beam)
        beam_focused = beam_profile_focus(mask_param_for_beam)
        beam_profile = beam_section(beam_focused, beam_3d_sections_filepath, args.x_min, args.x_max,args.y_min,args.y_max)
        beam_profile_out_path = os.path.join(out_dir, "beam_profile.tiff")
        skimage.io.imsave(beam_profile_out_path, (beam_profile/1e6).astype(np.uint16))
        print(f"Saved beam profile for input mask to {beam_profile_out_path}")

    # Run inference for each selected key
    for batch in selected_keys:
        data = labels_dict[batch]
        xyz_np = data['xyz']
        xyz = torch.tensor(xyz_np, dtype=torch.float32).to(config['device'])
        xyz_between_beads = data['xyz_between_beads']

        # Physical layer inference
        phys_img = run_inference(phys_model, mask_param, xyz)
        img_save_tiff(phys_img[display_batch], out_dir, "learned_mask_camera", batch)

        # CNN layer inference
        num_classes = config['num_classes']
        if num_classes == 1:
            cnn_img = run_inference(cnn_model, mask_param, xyz, 'cnn')
            img_save_tiff(cnn_img[0], out_dir, "inference_cnn", batch)
        if num_classes == 3:
            cnn_img = run_inference(cnn_model, mask_param, xyz, 'cnn_3_class')
            img_save_tiff(cnn_img[0], out_dir, "inference_cnn", batch, True)

        # Save ground truth
        if num_classes == 1:
            gt_img = batch_xyz_to_boolean_grid(xyz_np, config)
            if torch.is_tensor(gt_img):
                gt_img = gt_img.detach().squeeze().cpu().numpy()
            img_save_tiff(gt_img[0].astype(np.uint8), out_dir, "ground_truth", batch)
            # Compute metrics for binary
            compute_and_log_metrics(gt_img[0], cnn_img[0], out_dir, f"batch_{batch}", num_classes=2)
        if num_classes == 3: 
            gt_img = batch_xyz_to_3_class_grid(xyz, xyz_between_beads, config)  
            gt_img = gt_img.squeeze(1)
            img_save_tiff(gt_img[0].astype(np.uint8), out_dir, "ground_truth", batch, True)
            # Compute metrics for 3-class
            compute_and_log_metrics(gt_img[0], cnn_img[0], out_dir, f"batch_{batch}", num_classes=3)

        # Empty mask inference
        if args.empty_mask:
            empty_mask_tensor = torch.zeros_like(mask_tensor)
            empty_mask_param = torch.nn.Parameter(empty_mask_tensor, requires_grad=False)
            empty_phys_img = run_inference(phys_model, empty_mask_param, xyz)
            img_save_tiff(empty_phys_img[display_batch], out_dir, "empty_mask_camera", batch)
            if num_classes == 1:
                empty_cnn_img = run_inference(cnn_model, empty_mask_param, xyz, 'cnn')
                img_save_tiff(empty_cnn_img[0], out_dir, "inference_empty_cnn", batch)
            if num_classes == 3:
                empty_cnn_img = run_inference(cnn_model, empty_mask_param, xyz, 'cnn_3_class')
                img_save_tiff(empty_cnn_img[0], out_dir, "inference_empty_cnn", batch, True)
            img_save_tiff(empty_phys_img[display_batch], out_dir, "empty_mask_camera", batch)

        if args.bead_volume:
            bead_vol = batch_xyz_to_3class_volume(xyz_np, xyz_between_beads, config)
            # Save all three classes in one image: 0=background, 127=between beads, 255=beads
            bead_vol_dir = os.path.join(out_dir, "bead_volume")
            os.makedirs(bead_vol_dir, exist_ok=True)
            save_3d_volume_as_tiffs(bead_vol, bead_vol_dir)

        # Paper mask inference
        if args.paper_mask:
            paper_mask_param = paper_mask_param.to(config['device'])
            paper_phys_img = run_inference(phys_model, paper_mask_param, xyz)
            img_save_tiff(paper_phys_img[display_batch], out_dir, "paper_mask_camera", batch)
            if num_classes == 1:
                paper_cnn_img = run_inference(cnn_model, paper_mask_param, xyz, 'cnn')
                img_save_tiff(paper_cnn_img[0], out_dir, "inference_paper_cnn", batch)
            if num_classes == 3:
                paper_cnn_img = run_inference(cnn_model, paper_mask_param, xyz, 'cnn_3_class')
                img_save_tiff(paper_cnn_img[0], out_dir, "inference_paper_cnn", batch, True)
            
            # Generate beam profile for paper mask
            if args.generate_beam_profile:
                paper_mask_np_for_beam = paper_mask_tensor.cpu().numpy()
                mask_real = phase_mask_gen()
                mask_param_for_beam = mask_real*np.exp(1j*paper_mask_np_for_beam)
                beam_focused = beam_profile_focus(mask_param_for_beam)
                paper_beam_3d_sections_filepath = beam_3d_sections_filepath + "_paper"
                if not os.path.exists(paper_beam_3d_sections_filepath):
                    os.makedirs(paper_beam_3d_sections_filepath)
                beam_profile = beam_section(beam_focused, paper_beam_3d_sections_filepath, args.x_min, args.x_max,args.y_min,args.y_max)
                beam_profile_out_path = os.path.join(out_dir, "beam_profile_paper_mask.tiff")
                skimage.io.imsave(beam_profile_out_path, (beam_profile/1e6).astype(np.uint16))
                print(f"Saved beam profile for paper mask to {beam_profile_out_path}")

if __name__ == "__main__":
    main()
