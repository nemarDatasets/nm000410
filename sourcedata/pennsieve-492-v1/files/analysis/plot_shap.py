"""
SHAP (SHapley Additive exPlanations) Analysis for SEEG Models

This module implements SHAP analysis to interpret model predictions and understand
which temporal/spectral features contribute most to SOZ classification.

Main Functions:
    compute_shap_values: Calculate SHAP values for model predictions
    plot_shap_summary: Create summary visualizations of feature importance
    plot_shap_dependence: Show feature interaction effects
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
from matplotlib.gridspec import GridSpec
import random
from scipy.ndimage import gaussian_filter1d
from seeg_classify_all import *
from multi_scale_ori import *
def clear_gpu_memory():
    """Clear GPU memory cache"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
class SHAPTimeWindowAnalyzer:
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
        
        clear_gpu_memory()  # Clear before computation
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        print("Preparing background data...")
        self.background_data = self.prepare_background_data()
        print(f"Background data shape: {self.background_data.shape}")
        
        os.makedirs('/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/pos_confidence', exist_ok=True)
        os.makedirs('/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/neg_confidence', exist_ok=True)
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
    def find_windows_by_confidence(self) -> Tuple[List[Dict], List[Dict]]:
        """Find windows with high confidence predictions for both classes"""
        pos_windows = []
        neg_windows = []
        fold = 'fold_5'
        thresh = self.results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        for fold_key, fold_data in self.results['fold_metrics'].items():
            if 'detailed_results' in fold_data:
                for patient, patient_data in fold_data['detailed_results'].items():
                    for location in patient_data['locations'].keys():
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                label = patient_data['locations'][location]['lead_combinations'][lead_combo]["true_label"]
                                prediction = float(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'] >= thresh)
                                correct = (prediction == label)
                                if correct:
                                    preds = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["predictions"]
                                    confs = patient_data['locations'][location]['lead_combinations'][lead_combo]["windows"]["confidences"]
                                    num_windows = len(preds)
                                    for i in range(num_windows):
                                        pred = preds[i]
                                        conf = confs[i]
                                        window_info = {
                                            'patient': patient,
                                            'leads': lead_combo.split("_"),
                                            'window_index': i,
                                            'confidence': conf,
                                            'prediction': pred,
                                            'true_label': label,
                                            'location':location
                                        }
                                        if label == 1:
                                            pos_windows.append(window_info)
                                        else:
                                            neg_windows.append(window_info)
                                        
        pos_windows.sort(key=lambda x: x['confidence'], reverse=True)
        neg_windows.sort(key=lambda x: x['confidence'], reverse=True)
        
        return pos_windows, neg_windows
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
    def plot_time_window_importance(self, window_info: Dict, save_path: str = None):
        """Plot signal with enhanced SHAP importance visualization"""
        window_data, shap_values = self.compute_shap_values(window_info)
        shap_values = np.squeeze(shap_values)
        
        total_samples = window_data.shape[1]
        time_points = np.arange(total_samples) / self.sample_rate
        
        all_smoothed_shaps = []
        
        for idx, (lead, lead_shap) in enumerate(zip(window_info['leads'], shap_values)):
            smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
            all_smoothed_shaps.append(smoothed_shap)
        
        global_min = min(np.min(shap) for shap in all_smoothed_shaps)
        global_max = max(np.max(shap) for shap in all_smoothed_shaps)
        
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
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(len(window_info['leads']), 1, figure=fig)
        gs.update(left=0.12, right=0.85, hspace=0.4)
        
        for i, (lead, data, smoothed_shap) in enumerate(zip(window_info['leads'], 
                                                        window_data,
                                                        all_smoothed_shaps)):
            ax = fig.add_subplot(gs[i])
            
            norm = plt.Normalize(vmin=-max(abs(global_min), abs(global_max)), 
                            vmax=max(abs(global_min), abs(global_max)))
            
            X, Y = np.meshgrid(time_points, np.linspace(-10, 10, 100))
            C = np.tile(smoothed_shap, (100, 1))
            
            pcm = ax.pcolormesh(X, Y, C, cmap='RdBu_r', norm=norm, alpha=0.3, shading='gouraud')
            
            ax.plot(time_points, data, 'black', linewidth=1.5, zorder=3)
            
            ax.set_xlim(time_points[0], time_points[-1])
            ax.set_ylim(-10, 10)
            ax.grid(True, which='major', linestyle='-', alpha=0.2, zorder=0)
            
            if i == len(window_info['leads']) - 1:
                ax.set_xlabel('Time (seconds)', fontsize=20, fontweight='bold')
                ax.tick_params(axis='x', which='major', length=3, width=3, labelsize=20)
            else:
                ax.set_xticklabels([])
            
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.tick_params(axis='y', which='major', length=3, width=3, labelsize=20)
            
        title = (f"Patient {window_info['patient']}\n"
                f"Confidence: {window_info['confidence']:.3f} | "
                f"Prediction: {window_info['prediction']:.3f} | "
                f"True Label: {window_info['true_label']}")
        
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        n_bins = 256
        colors = plt.cm.RdBu_r(np.linspace(0, 1, n_bins))
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        gradient = np.linspace(-1, 1, n_bins).reshape(-1, 1)
        gradient_img = np.hstack((gradient, gradient))
        gradient_img = np.flipud(gradient_img)  # Flip the image vertically
        
        cax.imshow(gradient_img, aspect='auto', cmap=custom_cmap)
        cax.set_xticks([])
        
        cax.set_yticks([0, 128, 255])
        cax.set_yticklabels([f'{global_max:.4f}',
                            '0',
                            f'{global_min:.4f}'],
                            fontsize=14,
                            fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
    def analyze_windows(self, n_windows: int = 3):
        """Analyze top N confidence windows for both positive and negative classes"""
        pos_windows, neg_windows = self.find_windows_by_confidence()
        print(f"Found {len(pos_windows)} positive windows above confidence threshold {self.conf_threshold}")
        print(f"Found {len(neg_windows)} negative windows above confidence threshold {self.conf_threshold}")
        random_pos_windows = random.sample(pos_windows, n_windows)
        for i, window in enumerate(random_pos_windows):
            print(f"\nAnalyzing positive window {i+1}/{n_windows}")
            print(f"Patient: {window['patient']}")
            print(f"Confidence: {window['confidence']:.3f}")
            
            save_path = f'/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/pos_confidence/window_{i}_pat_{window["patient"]}_conf_{window["confidence"]:.3f}_adjusted.png'
            self.plot_time_window_importance(window, save_path)
        
        random_neg_windows = random.sample(neg_windows, n_windows)
        for i, window in enumerate(random_neg_windows):
            print(f"\nAnalyzing negative window {i+1}/{n_windows}")
            print(f"Patient: {window['patient']}")
            print(f"Confidence: {window['confidence']:.3f}")
            
            save_path = f'/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/neg_confidence/window_{i}_pat_{window["patient"]}_conf_{window["confidence"]:.3f}_adjusted.png'
            self.plot_time_window_importance(window, save_path)
def main():
    model_path = "/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/model_states/fold_5_results_1218_perm4_alllocs_v2_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = OptimizedMSResNet(input_channel=4, num_classes=1, initial_channels=16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    splits = load_splits('/home/sameer/all_location_restingSEEG_SOZ/patient_cv_alllocs_4ormore_v2.json')
    counter = 1
    for train, val, test in splits:
        if counter == 5:
            total_pts = train + val + test
            train_pts = train
            val_pts = val
            test_pts = test
            break
        counter += 1
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
    
    results_path = "/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/results_json/complete_results_1218_perm4_alllocs_v2.json"
    analyzer = SHAPTimeWindowAnalyzer(model, dataset, dataloader, test_pts, results_path, device)
    
    analyzer.analyze_windows(n_windows=1)
if __name__ == "__main__":
    main()