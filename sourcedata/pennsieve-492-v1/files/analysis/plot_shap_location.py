"""
Location-Specific SHAP Analysis

This module extends SHAP analysis to examine how feature importance varies
across different brain anatomical locations.

Main Functions:
    compute_shap_by_location: Calculate SHAP values grouped by brain location
    plot_location_shap_comparison: Compare feature importance across locations
    analyze_location_specific_patterns: Identify location-specific biomarkers
    clear_gpu_memory: Utility to clear GPU memory cache
"""

import matplotlib
matplotlib.use('Agg')
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
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
    def __init__(self, model, dataset, dataloader, train_pts, results_path: str, 
                 device='cuda', conf_threshold=0.6, sample_rate=500):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.dataloader = dataloader
        self.train_pts = train_pts
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
   
    def find_windows_by_location(self) -> Dict[str, List[Dict]]:
        """Find windows with highest confidence predictions for each location"""
        location_windows = {}
        fold = 'fold_5'
        thresh = self.results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        
        for fold_key, fold_data in self.results['fold_metrics'].items():
            if 'detailed_results' in fold_data:
                for patient, patient_data in fold_data['detailed_results'].items():
                    for location in patient_data['locations'].keys():
                        if location not in location_windows:
                            location_windows[location] = []
                            
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                label = patient_data['locations'][location]['lead_combinations'][lead_combo]["true_label"]
                                prediction = float(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'] >= thresh)
                                correct = (prediction == label)
                                
                                if correct and label == 1:
                                    preds = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["predictions"]
                                    confs = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["confidences"]
                                    
                                    for i in range(len(preds)):
                                        window_info = {
                                            'patient': patient,
                                            'leads': lead_combo.split("_"),
                                            'window_index': i,
                                            'confidence': confs[i],
                                            'prediction': preds[i],
                                            'true_label': label,
                                            'location': location
                                        }
                                        location_windows[location].append(window_info)
        
        for location in location_windows:
            location_windows[location].sort(key=lambda x: x['confidence'], reverse=True)
            
        return location_windows
    def compute_shap_values(self, window_info: Dict):
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
                
            return window_data, shap_values
            
        except Exception as e:
            print(f"Error computing SHAP values: {str(e)}")
            raise
    
    def plot_location_importance(self, location_set: List[str], save_path: str):
        """Plot most confident windows from each location in a grid"""
        location_windows = self.find_windows_by_location()
        
        location_data = {}
        for loc in location_set:
            if loc in location_windows and location_windows[loc]:
                total_windows = len(location_windows[loc])
                n_top_windows = max(1, int(total_windows * 0.1))
                
                selected_window = random.choice(location_windows[loc][:n_top_windows])
                location_data[loc] = selected_window
                print(f"Selected window with confidence {selected_window['confidence']:.3f} "
                        f"from top {n_top_windows} windows")
        n_locations = len(location_data)
        n_cols = 4  # Set to 4 columns
        n_rows = (n_locations + n_cols - 1) // n_cols
        
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#CCCCCC',
            'font.size': 12,
            'font.family': 'sans-serif',
            'axes.linewidth': 4,
            'axes.labelsize': 13,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
        })
        
        fig = plt.figure(figsize=(30, 8*n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        gs.update(hspace=0.30, wspace=0.15)  # Reduced wspace from 0.3 to 0.15
        
        all_data = {}
        global_min = float('inf')
        global_max = float('-inf')
        
        for i, (loc, window_info) in enumerate(location_data.items()):
            try:
                window_data, shap_values = self.compute_shap_values(window_info)
                shap_values = np.squeeze(shap_values)
                
                smoothed_shaps = []
                for lead_shap in shap_values:
                    smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
                    smoothed_shaps.append(smoothed_shap)
                    global_min = min(global_min, np.min(smoothed_shap))
                    global_max = max(global_max, np.max(smoothed_shap))
                
                all_data[loc] = {
                    'window_data': window_data,
                    'smoothed_shaps': smoothed_shaps,
                    'window_info': window_info
                }
            except Exception as e:
                print(f"Error processing location {loc}: {str(e)}")
                continue
        
        max_abs_val = max(abs(global_min), abs(global_max))
        linthresh = max_abs_val * 1e-1
        norm = SymLogNorm(linthresh=linthresh,
                        linscale=1.0,
                        vmin=-max_abs_val,
                        vmax=max_abs_val,
                        base=10)
        
        for i, (loc, data) in enumerate(all_data.items()):
            row = i // n_cols
            col = i % n_cols
            
            window_data = data['window_data']
            smoothed_shaps = data['smoothed_shaps']
            
            nested_gs = gs[row, col].subgridspec(len(window_data), 1, hspace=0.2)
            
            time_points = np.arange(window_data.shape[1]) / self.sample_rate
            
            for j, (lead_data, smoothed_shap) in enumerate(zip(window_data, smoothed_shaps)):
                ax = fig.add_subplot(nested_gs[j])
                
                X, Y = np.meshgrid(time_points, np.linspace(-10, 10, 100))
                C = np.tile(smoothed_shap, (100, 1))
                
                pcm = ax.pcolormesh(X, Y, C, cmap='RdBu_r', norm=norm, alpha=0.3, shading='gouraud')
                
                ax.plot(time_points, lead_data, 'black', linewidth=1.0, zorder=3)
                
                ax.set_xlim(time_points[0], time_points[-1])
                ax.set_ylim(-10, 10)
                ax.grid(True, which='major', linestyle='-', alpha=0.2)
                
                if j == 0:
                    ax.set_title(loc, fontsize=14, fontweight='bold', pad=10)
                
                if j != len(window_data) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel('Time (seconds)', fontsize=12)
                
                ax.set_yticks([-10, 0, 10])
                ax.tick_params(axis='y', which='major', length=3, width=3, labelsize=10)
        
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        
        cax.yaxis.set_label_position('right')
        cax.yaxis.tick_right()
        
        tick_locations = np.array([-max_abs_val, -linthresh, 0, linthresh, max_abs_val])
        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels([f'{val:.1e}' for val in tick_locations])
        cbar.ax.tick_params(labelsize=10, length=6, width=1)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
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
    dataset = SPES_Comb(test_pts, 
                       random_val=0,
                       try_all=True, 
                       num_files=50, 
                       augment=True, 
                       window=15000, 
                       stride=9000, 
                       normalize_method="robust",
                       size_permutation=4)
    dataloader = PatientDataLoader(dataset, batch_size=64)
    
    analyzer = EnhancedSHAPTimeWindowAnalyzer(model, dataset, dataloader, train_pts, results_path, device)
    
    location_set = [
        "RIGHT-HIPPOCAMPUS",
        "LEFT-HIPPOCAMPUS"
    ]
    
    save_path = '/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/location_analysis/all_locations_analysis_v5.png'
    analyzer.plot_location_importance(location_set, save_path)
if __name__ == "__main__":
    main()