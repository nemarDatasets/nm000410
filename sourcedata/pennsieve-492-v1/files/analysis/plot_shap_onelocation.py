"""
Single Location Deep-Dive SHAP Analysis

This module provides detailed SHAP analysis for individual brain locations,
enabling focused investigation of location-specific classification patterns.

Main Functions:
    analyze_single_location: Comprehensive SHAP analysis for one location
    plot_temporal_shap_evolution: Show how features evolve over time
    identify_critical_timepoints: Find most discriminative temporal windows
    clear_gpu_memory: Utility to clear GPU memory cache
"""

from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import SymLogNorm
from seeg_classify_all import *
from multi_scale_ori import *
def clear_gpu_memory():
    """Clear GPU memory cache"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
class EnhancedSHAPTimeWindowAnalyzer:
    def __init__(self, model, dataset, dataloader, train_pts, test_pts, results_path: str, 
                 device='cuda', conf_threshold=0.6, sample_rate=500):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.dataloader = dataloader
        self.train_pts = train_pts
        self.test_pts = test_pts
        self.conf_threshold = conf_threshold
        self.sample_rate = sample_rate
        self.model.eval()
        
        clear_gpu_memory()
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        print("Preparing background data...")
        self.background_data = self.prepare_background_data()
        print(f"Background data shape: {self.background_data.shape}")
        
        os.makedirs('/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/location_analysis', exist_ok=True)
    def prepare_background_data(self, num_background: int = 1000):
        """Prepare background data from training dataset"""
        background_data = []
        background_loader = self.dataloader.get_patient_loader(self.train_pts)
        
        with torch.no_grad():
            for data, _, _, _, _, _ in background_loader:
                background_data.append(data)
                if len(background_data) * data.shape[0] >= num_background:
                    break
                
        background_data = torch.cat(background_data)[:num_background]
        return background_data.to(self.device)
    def find_patient_windows(self, loc) -> Dict[str, Dict[str, List[Dict]]]:
        """Find windows for each patient, organized by location"""
        patient_location_windows = defaultdict(lambda: defaultdict(list))
        fold = 'fold_5'  # Assuming we're using fold 5
        thresh = self.results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        
        for fold_key, fold_data in self.results['fold_metrics'].items():
            if 'detailed_results' in fold_data:
                for patient, patient_data in fold_data['detailed_results'].items():
                    if patient not in self.test_pts:
                        continue
                        
                    has_loc = any(loc in loc_key for loc_key in patient_data['locations'].keys())
                    if not has_loc:
                        continue
                        
                    for location in patient_data['locations'].keys():
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                label = patient_data['locations'][location]['lead_combinations'][lead_combo]["true_label"]
                                prediction = float(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'] >= thresh)
                                correct = (prediction == label)
                                
                                if correct and label == 1:
                                    preds = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["predictions"]
                                    confs = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["confidences"]
                                    
                                    max_conf = 0
                                    index_to_use = 0
                                    for i in range(len(preds)):
                                        if confs[i] > max_conf:
                                            max_conf = confs[i]
                                            index_to_use = i
                                    
                                        window_info = {
                                            'patient': patient,
                                            'leads': lead_combo.split("_"),
                                            'window_index': i,
                                            'confidence': confs[i],
                                            'prediction': preds[i],
                                            'true_label': label,
                                            'location': location
                                        }
                                    patient_location_windows[patient][location].append(window_info)
        
        for patient in patient_location_windows:
            for location in patient_location_windows[patient]:
                patient_location_windows[patient][location].sort(
                    key=lambda x: x['confidence'], 
                    reverse=True
                )
                
        return patient_location_windows
    def compute_shap_values(self, window_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute SHAP values for a specific window"""
        patient = window_info['patient']
        leads = window_info['leads']
        window_start = window_info['window_index'] * self.dataset.stride
        location = window_info['location']
        
        metadata = self.dataset._get_patient_metadata(patient)
        lead_indices = [metadata['locations'][location]['leads'].index(lead) for lead in leads]
        mmap_data = self.dataset.get_mmap(patient)
        data = mmap_data[location]
        window_data = data[lead_indices, window_start:window_start+self.dataset.window]        
        
        window_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
        
        try:
            explainer = shap.GradientExplainer(self.model, self.background_data)
            shap_values = explainer.shap_values(window_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if torch.is_tensor(shap_values):
                shap_values = shap_values.cpu().numpy()
                
            return window_data, shap_values
            
        except Exception as e:
            print(f"Error computing SHAP values: {str(e)}")
            return None
            
    def compute_patient_bounds(self, patient_windows: Dict[str, List[Dict]], loc) -> Tuple[float, float]:
        """Compute min/max SHAP values across all locations for a patient"""
        global_min = float('inf')
        global_max = float('-inf')
        found_valid_data = False
        for location in patient_windows:
            windows = patient_windows[location]
            n_top_windows = max(1, int(len(windows) * 0.1))
            top_windows = windows[:n_top_windows]
            if loc in location:
                for window_info in top_windows:
                    result = self.compute_shap_values(window_info)
                    if result is not None:
                        _, shap_values = result
                        for lead_shap in shap_values:
                            smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
                            global_min = min(global_min, np.min(smoothed_shap))
                            global_max = max(global_max, np.max(smoothed_shap))
                        found_valid_data = True
                    
                    torch.cuda.empty_cache()
        if not found_valid_data:
            return None, None
            
        return global_min, global_max
    
    def normalize_shap_values(self, shap_values: np.ndarray, global_min: float, global_max: float) -> np.ndarray:
        """Normalize SHAP values using patient-specific bounds"""
        if global_min == global_max:
            return np.zeros_like(shap_values)
            
        normalized = (shap_values - global_min) / (global_max - global_min)
        return normalized
    
    def plot_importance(self, location, save_path: str):
        """Plot location windows with temporal SHAP importance"""
        print(f"Starting plotting for {location}...")
        patient_windows = self.find_patient_windows(location)
        
        print("Computing global bounds for visualization...")
        super_global_min = float('inf')
        super_global_max = float('-inf')
        
        plot_data = {}
        for patient, location_windows in tqdm(patient_windows.items()):
            min_val, max_val = self.compute_patient_bounds(location_windows, location)
            if min_val is None or max_val is None:
                print(f"Skipping patient {patient}: Could not compute bounds")
                continue
                
            super_global_min = min(super_global_min, min_val)
            super_global_max = max(super_global_max, max_val)
            
            for loc_name, windows in location_windows.items():
                if location in loc_name:  # Changed this line
                    if windows:
                        n_top_windows = max(1, int(len(windows) * 0.1))
                        selected_window = random.choice(windows[:n_top_windows])
                        plot_data[f"{patient}-{loc_name}"] = {
                            'window_info': selected_window,
                            'bounds': (min_val, max_val)  # Patient-specific bounds
                        }
            
        n_windows = len(plot_data)
        n_cols = 4
        n_rows = (n_windows + n_cols - 1) // n_cols
        if n_windows > 5:
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'grid.color': '#CCCCCC',
                'font.size': 12,
                'font.family': 'sans-serif',
                'axes.linewidth': 2,  # Increased from 1
                'axes.labelsize': 14,  # Increased
                'xtick.labelsize': 12,  # Increased
                'ytick.labelsize': 12,  # Increased
                'font.weight': 'bold'  # Added for bold text
            })
            fig = plt.figure(figsize=(30, 8*n_rows))
            gs = GridSpec(n_rows, n_cols, figure=fig)
            gs.update(hspace=0.30, wspace=0.15)
            
            max_abs_val = max(abs(super_global_min), abs(super_global_max))
            print(f"max absolute val = {max_abs_val}")
            
            linthresh = max_abs_val * 1e-4
            norm = SymLogNorm(linthresh=linthresh,
                            linscale=1.0,
                            vmin=-max_abs_val,
                            vmax=max_abs_val,
                            base=10)
            
            print(f"Plotting {n_windows} windows...")
            for i, (key, data) in enumerate(plot_data.items()):
                try:
                    print(f"Processing window {i+1}/{n_windows}: {key}")
                    window_info = data['window_info']
                    patient_min, patient_max = data['bounds']
                    
                    window_data, shap_values = self.compute_shap_values(window_info)
                    if window_data is None:
                        continue
                        
                    shap_values = np.squeeze(shap_values)
                    
                    row = i // n_cols
                    col = i % n_cols
                    nested_gs = gs[row, col].subgridspec(len(window_data), 1, hspace=0.2)
                    
                    time_points = np.arange(window_data.shape[1]) / self.sample_rate
                    X, Y = np.meshgrid(time_points, np.linspace(-10, 10, 50))
                    
                    for j, (lead_data, lead_shap) in enumerate(zip(window_data, shap_values)):
                        ax = fig.add_subplot(nested_gs[j])
                        
                        smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
                        
                        C = np.tile(smoothed_shap, (50, 1))
                        
                        pcm = ax.pcolormesh(X, Y, C, cmap='RdBu_r', 
                                        norm=norm, alpha=0.8, 
                                        shading='gouraud')  # Keep gouraud for smoothness
                        
                        ax.plot(time_points, lead_data, 'black', linewidth=1, zorder=3)
                        
                        ax.set_xlim(time_points[0], time_points[-1])
                        ax.set_ylim(-10, 10)
                        ax.grid(True, which='major', linestyle='-', alpha=0.2)
                        
                        if j == 0:
                            title = f"{window_info['location']}-{window_info['patient']}"
                            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                        
                        if j != len(window_data) - 1:
                            ax.set_xticklabels([])
                        else:
                            ax.set_xlabel('Time (seconds)', fontsize=12)
                        
                        ax.set_yticks([-10, 0, 10])
                        ax.tick_params(axis='y', which='major', length=4, width=2, labelsize=10)  # Increased width and length
                    
                    del shap_values, window_data
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing window {key}: {str(e)}")
                    continue
            
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            
            cax.yaxis.set_label_position('right')
            cax.yaxis.tick_right()
            tick_locations = np.array([-max_abs_val, -max_abs_val/2, 0, max_abs_val/2, max_abs_val])
            cbar.set_ticks(tick_locations)
            cbar.set_ticklabels([f'{val:.2e}' for val in tick_locations])
            cbar.ax.tick_params(labelsize=12, length=8, width=2)  # Increased parameters
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close('all')
def main():
    model_path = "/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/model_states/fold_5_results_1218_perm4_alllocs_v2_model.pth"
    results_path = "/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/results_json/complete_results_1218_perm4_alllocs_v2.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = OptimizedMSResNet(input_channel=4, num_classes=1, initial_channels=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    splits = load_splits('/home/sameer/all_location_restingSEEG_SOZ/patient_cv_alllocs_4ormore_v2.json')
    for i, (train, val, test) in enumerate(splits, 1):
        if i == 5:
            total_pts = train + val + test
            train_pts = train
            test_pts = test
            break
    dataset = SPES_Comb(total_pts, 
                       random_val=0,
                       try_all=True, 
                       num_files=50, 
                       augment=True, 
                       window=15000, 
                       stride=9000, 
                       normalize_method="robust",
                       size_permutation=4)
    dataloader = PatientDataLoader(dataset, batch_size=64)
    
    analyzer = EnhancedSHAPTimeWindowAnalyzer(model, dataset, dataloader, train_pts, test_pts, results_path, device)
    
    location_set = ["CTX-LH-FUSIFORM",
                "CTX-LH-INFERIORTEMPORAL",
                "CTX-LH-LATERALORBITOFRONTAL",
                "CTX-LH-MIDDLETEMPORAL",
                "CTX-LH-PRECENTRAL",
                "CTX-LH-SUPERIORTEMPORAL",
                "CTX-LH-TEMPORALPOLE",
                "LEFT-HIPPOCAMPUS",
                "CTX-LH-INSULA",
                "LEFT-AMYGDALA",
                "RIGHT-HIPPOCAMPUS",
                "RIGHT-AMYGDALA",
                "CTX-RH-MIDDLETEMPORAL",
                "CTX-RH-LATERALORBITOFRONTAL",
                "CTX-RH-INFERIORTEMPORAL",
                "CTX-RH-SUPERIORFRONTAL",
                "CTX-RH-INSULA",
                "CTX-RH-MEDIALORBITOFRONTAL",
                "CTX-RH-SUPRAMARGINAL",
                "CTX-RH-TEMPORALPOLE",
                "CTX-LH-LATERALOCCIPITAL",
                "CTX-RH-CAUDALMIDDLEFRONTAL",
                "CTX-RH-POSTCENTRAL",
                "CTX-LH-INFERIORPARIETAL",
                "CTX-RH-ISTHMUSCINGULATE"]
                
    locations_to_use = list(set([loc.split("-")[-1] for loc in location_set]))
  
    for loc in locations_to_use:
        save_path = f'/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/location_analysis/{loc}_locations_analysis_combinedfigure.png'
        analyzer.plot_importance(loc, save_path)
if __name__ == "__main__":
    main()