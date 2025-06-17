import argparse
from CGS_tools import *
import warnings


def main():
    """Main execution function."""
    warnings.filterwarnings("ignore")

    # -------- Argument Parsing --------
    parser = argparse.ArgumentParser(description="Superhuman Model Inference with CGS Patch Selection")
    parser.add_argument('-c', '--cfg', type=str, default='SL', help='Name of YAML config (without extension)')
    args = parser.parse_args()

    # -------- Load Config & Device --------
    cfg = load_config(args.cfg)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # -------- Load Model --------
    checkpoint_path = cfg.MODEL.model_pth
    model = load_model(cfg, checkpoint_path, device)

    # -------- Load Validation Data --------
    valid_provider = ProviderValid(cfg)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, shuffle=False, num_workers=0)

    # -------- Inference --------
    features, positions = run_inference(model, val_loader, valid_provider, device)
    
    # -------- CGS Parameters --------
    subvolume_num = cfg.CGS.subvolume_num
    n_neighbors_list = cfg.CGS.n_neighbors_list
    window_size = cfg.CGS.window_size
    point_cloud_size = [valid_provider.num_z, valid_provider.num_y, valid_provider.num_x]

    # -------- CGS Patch Selection --------
    save_results(n_neighbors_list, subvolume_num, features, positions, window_size, point_cloud_size)
    
    # save subvolume
    stride = cfg.MODEL.stride
    patch_size = cfg.MODEL.crop_size
    volume = valid_provider.datasets[0]

    extract_subvolume(cfg, window_size, stride, patch_size, volume)

if __name__ == "__main__":
    main()
