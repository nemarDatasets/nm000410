"""
Multi-Model SHAP Comparison and Ensemble Analysis

This module implements SHAP analysis across multiple model architectures
and training configurations to identify robust features.

Main Functions:
    compute_multimodel_shap: Calculate SHAP values across multiple models
    plot_model_comparison: Compare feature importance across models
    identify_consensus_features: Find features consistently important across models
    ensemble_shap_analysis: Aggregate SHAP values for ensemble interpretation

Classes:
    ModelConfig: Configuration dataclass for model specifications
"""

from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, NamedTuple
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import SymLogNorm
from dataclasses import dataclass
from seeg_classify_all import *
from multi_scale_ori import *
@dataclass
class ModelConfig:
    model_path: str
    results_path: str
    fold: str
    normalize_method: str
def clear_gpu_memory():
    """Clear GPU memory cache"""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
class SHAPEqualizer:
    def __init__(self, shap_values, n_bins=128):
        self.n_bins = n_bins
        self.target_range = (-1, 1)
    
    def transform(self, shap_values):
        equalized = np.zeros_like(shap_values)
        
        for i in range(shap_values.shape[0]):
            lead_data = shap_values[i, :]
            
            pos_mask = lead_data > 0
            neg_mask = lead_data < 0
            
            if np.any(pos_mask):
                pos_data = lead_data[pos_mask]
                hist_pos, bin_edges_pos = np.histogram(pos_data, self.n_bins)
                cdf_pos = np.cumsum(hist_pos)
                cdf_pos = cdf_pos / cdf_pos[-1]  # Normalize to [0, 1]
                equalized[i, pos_mask] = np.interp(pos_data, bin_edges_pos[:-1], cdf_pos) * self.target_range[1]
            
            if np.any(neg_mask):
                neg_data = np.abs(lead_data[neg_mask])
                hist_neg, bin_edges_neg = np.histogram(neg_data, self.n_bins)
                cdf_neg = np.cumsum(hist_neg)
                cdf_neg = cdf_neg / cdf_neg[-1]  # Normalize to [0, 1]
                equalized[i, neg_mask] = -np.interp(neg_data, bin_edges_neg[:-1], cdf_neg) * self.target_range[1]
            
            zero_mask = lead_data == 0
            equalized[i, zero_mask] = 0
        
        print("\nEqualization check:")
        print(f"Input range: {np.min(shap_values):.2e} to {np.max(shap_values):.2e}")
        print(f"Output range: {np.min(equalized):.2e} to {np.max(equalized):.2e}")
        
        return equalized
    
class MultiModelSHAPAnalyzer:
    def __init__(self, model_configs: List[ModelConfig], device='cuda', 
                 conf_threshold=0.6, sample_rate=500, stats_only = False,
                 target_regions=None):
        self.device = device
        self.conf_threshold = conf_threshold
        self.sample_rate = sample_rate
        self.model_configs = model_configs
        self.target_regions = target_regions or ["HIPPOCAMPUS", "AMYGDALA", "INSULA"]
        
        self.models = []
        self.datasets = []
        self.dataloaders = []
        self.results = []
        self.train_pts = []
        self.test_pts = []
        self.background_data = []
        
        for config in model_configs:
            model, dataset, dataloader, train_pts, test_pts, results = self._initialize_model(config)
            self.models.append(model)
            self.datasets.append(dataset)
            self.dataloaders.append(dataloader)
            self.results.append(results)
            self.train_pts.append(train_pts)
            self.test_pts.append(test_pts)
            
        if not stats_only:
            print("Preparing background data...")
            for i, (dataset, dataloader, train_pts) in enumerate(zip(self.datasets, self.dataloaders, self.train_pts)):
                bg_data = self._prepare_background_data(dataset, dataloader, train_pts)
                self.background_data.append(bg_data)
                print(f"Model {i+1} background data shape: {bg_data.shape}")
            
            print("\nCreating global equalizers for SHAP values...")
            self.model_equalizers = []
            for model_idx in range(len(self.models)):
                print(f"\nCollecting SHAP values for model {model_idx}")
                shap_values = self.collect_model_shap_values(model_idx, n_samples=200)  # Increased samples
                print(f"Model {model_idx} raw SHAP range: {np.min(shap_values):.2e} to {np.max(shap_values):.2e}")
                
                equalizer = SHAPEqualizer(shap_values)
                self.model_equalizers.append(equalizer)
    def _initialize_model(self, config: ModelConfig):
        """Initialize model and associated components"""
        device = torch.device(self.device)
        checkpoint = torch.load(config.model_path, map_location=device)
        model = OptimizedMSResNet(input_channel=4, num_classes=1, initial_channels=16).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open(config.results_path, 'r') as f:
            results = json.load(f)
        
        splits = load_splits('/home/sameer/all_location_restingSEEG_SOZ/patient_cv_alllocs_4ormore_v2.json')
        for i, (train, val, test) in enumerate(splits, 1):
            if i == int(config.fold.split('_')[1]):
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
                           normalize_method=config.normalize_method,
                           size_permutation=4)
        dataloader = PatientDataLoader(dataset, batch_size=64)
        
        return model, dataset, dataloader, train_pts, test_pts, results
    def _prepare_background_data(self, dataset, dataloader, train_pts, num_background: int = 1000):
        """Prepare background data for SHAP analysis"""
        background_data = []
        background_loader = dataloader.get_patient_loader(train_pts)
        
        with torch.no_grad():
            for data, _, _, _, _, _ in background_loader:
                background_data.append(data)
                if len(background_data) * data.shape[0] >= num_background:
                    break
                
        background_data = torch.cat(background_data)[:num_background]
        return background_data.to(self.device)
        
    def find_case_windows(self, results, test_pts, region=None):
        """Find windows for each case type (TP, FP, TN, FN) optionally filtered by region"""
        cases = {
            'TP': [], 'FP': [], 'TN': [], 'FN': []
        }
        
        fold = self.model_configs[0].fold
        thresh = results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        
        for fold_key, fold_data in results['fold_metrics'].items():
            if 'detailed_results' in fold_data:
                for patient, patient_data in fold_data['detailed_results'].items():
                    if patient not in test_pts:
                        continue
                        
                    for loc_name in patient_data['locations'].keys():
                        if region is not None and region not in loc_name:
                            continue
                            
                        if 'lead_combinations' in patient_data['locations'][loc_name]:
                            for lead_combo in patient_data['locations'][loc_name]['lead_combinations'].keys():
                                label = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["true_label"]
                                prediction = float(patient_data['locations'][loc_name]['lead_combinations'][lead_combo]['prediction'] >= thresh)
                                
                                if label == 1 and prediction == 1:
                                    case_type = 'TP'
                                elif label == 0 and prediction == 1:
                                    case_type = 'FP'
                                elif label == 0 and prediction == 0:
                                    case_type = 'TN'
                                else:
                                    case_type = 'FN'
                                
                                preds = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["windows"]["predictions"]
                                confs = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["windows"]["confidences"]
                                
                                for i in range(len(preds)):
                                    window_info = {
                                        'patient': patient,
                                        'leads': lead_combo.split("_"),
                                        'window_index': i,
                                        'confidence': confs[i],
                                        'prediction': preds[i],
                                        'true_label': label,
                                        'location': loc_name
                                    }
                                    cases[case_type].append(window_info)
        
        for case_type in cases:
            cases[case_type].sort(key=lambda x: x['confidence'], reverse=True)
        
        return cases
    
    def collect_model_shap_values(self, model_idx: int, n_samples=100):
        all_shap_values = []
        all_regions = set()  # Track regions we sample from
        
        cases = self.find_case_windows(self.results[model_idx], self.test_pts[model_idx])
        total_samples = 0
        
        region_stats = defaultdict(lambda: defaultdict(int))
        
        for case_type, windows in cases.items():
            if windows:
                n_case_samples = min(n_samples // 4, len(windows))
                sampled_windows = random.sample(windows, n_case_samples)
                
                for window_info in sampled_windows:
                    result = self.compute_shap_values(model_idx, window_info)
                    if result is not None:
                        _, shap_values = result
                        smoothed_shap = gaussian_filter1d(shap_values, sigma=100, axis=1)
                        all_shap_values.append(smoothed_shap)
                        all_regions.add(window_info['location'])
                        region_stats[window_info['location']][case_type] += 1
                        total_samples += 1
        
        print(f"\nModel {model_idx} SHAP collection stats:")
        print(f"Total samples: {total_samples}")
        print(f"Unique regions: {len(all_regions)}")
        for region, case_counts in region_stats.items():
            print(f"\n{region}:")
            for case_type, count in case_counts.items():
                print(f"  {case_type}: {count} samples")
        
        if not all_shap_values:
            raise ValueError("No SHAP values collected!")
        
        concatenated = np.concatenate(all_shap_values, axis=1)
        print(f"\nSHAP value statistics:")
        print(f"Shape: {concatenated.shape}")
        print(f"Raw range: {np.min(concatenated):.2e} to {np.max(concatenated):.2e}")
        print(f"Mean abs: {np.mean(np.abs(concatenated)):.2e}")
        return concatenated
    
    def normalize_model_shaps(self, shap_values, model_idx):
        """Normalize SHAP values within a model to [-1, 1] range using global max for that model"""
        max_magnitude = np.max(np.abs(shap_values))
        print(f"\nModel {model_idx} max SHAP magnitude: {max_magnitude:.2e}")
        
        normalized_shaps = shap_values / max_magnitude
        print(f"Model {model_idx} normalized SHAP range: {np.min(normalized_shaps):.2e} to {np.max(normalized_shaps):.2e}")
        
        return normalized_shaps
    
    def create_optimized_plot(self, ax, time_points, lead_data, lead_shap, norm, y_lim=(-10.5, 10.5), y_ticks=(-10, 0, 10)):
        """
        Create an optimized plot with rasterized heatmap and vector overlays.
        Parameters:
            ax: matplotlib axis
            time_points: array of time points
            lead_data: signal data
            lead_shap: SHAP values
            norm: color normalization
            y_lim: tuple of y-axis limits
            y_ticks: tuple of y-axis tick locations
        """
        X, Y = np.meshgrid(time_points, np.linspace(y_lim[0], y_lim[1], 25))
        
        C = np.tile(lead_shap, (25, 1)).astype(np.float32)
        pcm = ax.pcolormesh(X, Y, C, 
                            cmap='coolwarm',
                            norm=norm,
                            alpha=0.75,
                            shading='nearest',  # More efficient than 'gouraud'
                            rasterized=True)    # Rasterize heatmap
        
        decimation = 1  # Plot every 5th point
        ax.plot(time_points[::decimation], 
                lead_data[::decimation], 
                'black', 
                linewidth=1.5, 
                zorder=3,
                rasterized=False)  # Keep as vector
        
        ax.set_ylim(y_lim)
        ax.set_yticks(y_ticks)
        ax.set_xlim(time_points[0], time_points[-1])
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
        
        return pcm
    def plot_region_cases(self, region: str, save_path: str, robust_only: bool = False):
        """Create a plot for one region showing all case types with optimized SVG output"""
        plt.rcParams['svg.fonttype'] = 'none'  # Keep fonts as text
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.weight'] = 'bold'
        
        windows_by_case = {}
        selected_windows = {}
        
        print(f"\nCollecting windows for {region}...")
        for model_idx, (results, test_pts) in enumerate(zip(self.results, self.test_pts)):
            if model_idx == 0:  # Only need to do this once
                cases = self.find_case_windows(results, test_pts, region)
                
                for case_type, windows in cases.items():
                    if windows:
                        top_20_percent = max(1, int(len(windows) * 0.2))
                        selected_windows[case_type] = random.choice(windows[:top_20_percent])
                        print(f"Found {case_type} window from patient {selected_windows[case_type]['patient']}")
                    else:
                        print(f"No {case_type} cases found for {region}")
        
        if not selected_windows:
            print(f"No windows found for region {region}")
            return
        
        n_cases = len(selected_windows)
        n_models = 1 if robust_only else len(self.models)
        
        fig = plt.figure(figsize=(24, 6*n_cases), dpi=150)  # Reduced DPI
        outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[25, 1])
        plot_gs = GridSpecFromSubplotSpec(n_cases, n_models, subplot_spec=outer_gs[0, 0],
                                        hspace=0.4, wspace=0.2)
        norm = plt.Normalize(vmin=-1, vmax=1)
        print("\nCreating visualizations...")
        for case_idx, (case_type, window_info) in enumerate(selected_windows.items()):
            for model_idx in range(n_models):
                if robust_only and model_idx == 1:
                    continue
                normalize_method = self.model_configs[model_idx].normalize_method
                print(f"Processing {case_type} for {normalize_method}")
                
                result = self.compute_shap_values(model_idx, window_info)
                if result is None:
                    continue
                    
                window_data, shap_values = result
                shap_values = np.squeeze(shap_values)
                
                smoothed_shap = gaussian_filter1d(shap_values, sigma=100, axis=1)
                equalized_shap = self.model_equalizers[model_idx].transform(smoothed_shap)
                
                pos_sum = np.sum(equalized_shap[equalized_shap > 0])
                neg_sum = np.abs(np.sum(equalized_shap[equalized_shap < 0]))
                total_sum = pos_sum + neg_sum
                pos_percent = (pos_sum / total_sum) * 100 if total_sum > 0 else 0
                neg_percent = (neg_sum / total_sum) * 100 if total_sum > 0 else 0
                nested_gs = GridSpecFromSubplotSpec(len(window_data), 1, 
                                                subplot_spec=plot_gs[case_idx, model_idx],
                                                hspace=0.2)
                
                time_points = np.arange(window_data.shape[1]) / self.sample_rate
                
                for lead_idx, (lead_data, lead_shap) in enumerate(zip(window_data, equalized_shap)):
                    ax = fig.add_subplot(nested_gs[lead_idx])
                    
                    y_lim = (-1.5, 1.5) if normalize_method == "histogram" else (-10.5, 10.5)
                    y_ticks = [-1, 0, 1] if normalize_method == "histogram" else [-10, 0, 10]
                    
                    pcm = self.create_optimized_plot(ax, time_points, lead_data, lead_shap, 
                                            norm, y_lim, y_ticks)
                    
                    if lead_idx == 0:
                        if normalize_method == "histogram":
                            title = f"{case_type}: {region}\nHistogram Equalization\n+{pos_percent:.1f}% / -{neg_percent:.1f}%"
                        else:
                            title = f"{case_type}: {region}\nMedian Normalization\n+{pos_percent:.1f}% / -{neg_percent:.1f}%"
                        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                    
                    if lead_idx == len(window_data) - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                        ax.tick_params(axis='x', which='major', length=4, width=2, labelsize=10)
                    else:
                        ax.set_xticklabels([])
                    
                    ax.tick_params(axis='y', which='major', length=4, width=2, labelsize=10)
                
                del shap_values, window_data
                torch.cuda.empty_cache()
        
        cax = fig.add_subplot(outer_gs[0, 1])
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.set_ylabel('Equalized SHAP Value', fontsize=16, fontweight='bold', 
                        rotation=-90, labelpad=20)
        
        tick_locations = [-1, -0.5, 0, 0.5, 1]
        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels([f'{val:g}' for val in tick_locations])
        cbar.ax.tick_params(labelsize=12, length=8, width=2)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='svg')
        plt.close('all')
        
    def plot_model_disagreements(self, save_path: str):
        """Plot cases with largest prediction differences where each model outperforms the other"""
        plt.rcParams['svg.fonttype'] = 'none'  # Keep fonts as text
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.weight'] = 'bold'
        
        print("\nFinding model disagreement examples...")
        
        robust_better_cases_pos = []
        robust_better_cases_neg = []
        hist_better_cases_pos = []
        hist_better_cases_neg = []
        
        robust_thresh = self.results[0]['fold_metrics'][self.model_configs[0].fold]['metrics']['lead_level']['optimal_threshold']
        hist_thresh = self.results[1]['fold_metrics'][self.model_configs[1].fold]['metrics']['lead_level']['optimal_threshold']
        
        for patient in self.test_pts[0]:
            robust_results = self.results[0]['fold_metrics'][self.model_configs[0].fold]['detailed_results'].get(patient, {})
            hist_results = self.results[1]['fold_metrics'][self.model_configs[1].fold]['detailed_results'].get(patient, {})
            
            if not robust_results or not hist_results:
                continue
                
            for location in robust_results.get('locations', {}):
                if location not in hist_results.get('locations', {}) or 'RH-MIDDLETEMPORAL' in location or 'RH-BANKSSTS' in location:
                    continue
                    
                robust_loc = robust_results['locations'][location]
                hist_loc = hist_results['locations'][location]
                
                for lead_combo in robust_loc.get('lead_combinations', {}):
                    if lead_combo not in hist_loc.get('lead_combinations', {}):
                        continue
                        
                    robust_data = robust_loc['lead_combinations'][lead_combo]
                    hist_data = hist_loc['lead_combinations'][lead_combo]
                    
                    true_label = robust_data['true_label']
                    robust_pred = float(robust_data['prediction'] >= robust_thresh)
                    hist_pred = float(hist_data['prediction'] >= hist_thresh)
                    
                    pred_diff = abs(robust_data['prediction'] - hist_data['prediction'])
                    
                    if robust_pred != hist_pred:
                        case_info = {
                            'patient': patient,
                            'location': location,
                            'leads': lead_combo.split("_"),
                            'true_label': true_label,
                            'robust_pred': robust_data['prediction'],
                            'hist_pred': hist_data['prediction'],
                            'pred_diff': pred_diff,
                            'window_info': None
                        }
                        
                        max_diff = -1
                        best_window_idx = 0
                        n_windows = min(len(robust_data["windows"]["predictions"]), 
                                    len(hist_data["windows"]["predictions"]))
                        
                        for i in range(n_windows):
                            window_diff = abs(robust_data["windows"]["predictions"][i] - 
                                        hist_data["windows"]["predictions"][i])
                            if window_diff > max_diff:
                                max_diff = window_diff
                                best_window_idx = i
                        
                        case_info['window_info'] = {
                            'patient': patient,
                            'leads': lead_combo.split("_"),
                            'window_index': best_window_idx,
                            'window_diff': max_diff,
                            'location': location
                        }
                        
                        if robust_pred == true_label and hist_pred != true_label:
                            if true_label == 1 and robust_pred > 0.5:
                                robust_better_cases_pos.append(case_info)
                            elif true_label == 0 and robust_pred < 0.5:
                                robust_better_cases_neg.append(case_info)
                        elif hist_pred == true_label and robust_pred != true_label:
                            if true_label == 1 and hist_pred > 0.5:
                                hist_better_cases_pos.append(case_info)
                            elif true_label == 0 and hist_pred < 0.5:
                                hist_better_cases_neg.append(case_info)
        
        print(f"Found {len(robust_better_cases_pos)} robust better positive cases")
        print(f"Found {len(robust_better_cases_neg)} robust better negative cases")
        print(f"Found {len(hist_better_cases_pos)} histogram better positive cases")
        print(f"Found {len(hist_better_cases_neg)} histogram better negative cases")
        
        if not all([robust_better_cases_pos, robust_better_cases_neg, 
                    hist_better_cases_pos, hist_better_cases_neg]):
            print("Not enough disagreement cases found!")
            return
        
        top_n = min(20, len(robust_better_cases_pos), len(robust_better_cases_neg),
                    len(hist_better_cases_pos), len(hist_better_cases_neg))
        
        selected_cases = [
            random.choice(robust_better_cases_pos[:top_n]),
            random.choice(robust_better_cases_neg[:top_n]),
            random.choice(hist_better_cases_pos[:top_n]),
            random.choice(hist_better_cases_neg[:top_n])
        ]
        
        n_cases = len(selected_cases)
        n_models = len(self.models)
        
        fig = plt.figure(figsize=(24, 6*n_cases), dpi=150)  # Reduced DPI for optimization
        outer_gs = GridSpec(1, 2, figure=fig, width_ratios=[25, 1])
        plot_gs = GridSpecFromSubplotSpec(n_cases, n_models, subplot_spec=outer_gs[0, 0],
                                        hspace=0.4, wspace=0.2)
        norm = plt.Normalize(vmin=-1, vmax=1)
        print("\nCreating visualizations...")
        for case_idx, case_info in enumerate(selected_cases):
            for model_idx in range(n_models):
                normalize_method = self.model_configs[model_idx].normalize_method
                winner = "Robust Better" if case_idx < 2 else "Histogram Better"
                
                result = self.compute_shap_values(model_idx, case_info['window_info'])
                if result is None:
                    continue
                    
                window_data, shap_values = result
                shap_values = np.squeeze(shap_values)
                
                smoothed_shap = gaussian_filter1d(shap_values, sigma=100, axis=1)
                equalized_shap = self.model_equalizers[model_idx].transform(smoothed_shap)
                
                pos_sum = np.sum(equalized_shap[equalized_shap > 0])
                neg_sum = np.abs(np.sum(equalized_shap[equalized_shap < 0]))
                total_sum = pos_sum + neg_sum
                pos_percent = (pos_sum / total_sum) * 100 if total_sum > 0 else 0
                neg_percent = (neg_sum / total_sum) * 100 if total_sum > 0 else 0
                nested_gs = GridSpecFromSubplotSpec(len(window_data), 1, 
                                                subplot_spec=plot_gs[case_idx, model_idx],
                                                hspace=0.2)
                
                time_points = np.arange(window_data.shape[1]) / self.sample_rate
                
                for lead_idx, (lead_data, lead_shap) in enumerate(zip(window_data, equalized_shap)):
                    ax = fig.add_subplot(nested_gs[lead_idx])
                    
                    y_lim = (-1.5, 1.5) if normalize_method == "histogram" else (-10.5, 10.5)
                    y_ticks = [-1, 0, 1] if normalize_method == "histogram" else [-10, 0, 10]
                    
                    pcm = self.create_optimized_plot(ax, time_points, lead_data, lead_shap, 
                                            norm, y_lim, y_ticks)
                    
                    if lead_idx == 0:
                        model_pred = case_info['robust_pred'] if model_idx == 0 else case_info['hist_pred']
                        case_diff = case_info['pred_diff']
                        title = (f"{winner}: {case_info['location']}\n"
                                f"{'Histogram' if normalize_method == 'histogram' else 'Median'} "
                                f"Normalization (pred: {model_pred:.3f}, true: {case_info['true_label']}, "
                                f"diff: {case_diff:.3f})\n"
                                f"+{pos_percent:.1f}% / -{neg_percent:.1f}%")
                        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
                    
                    if lead_idx == len(window_data) - 1:
                        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                        ax.tick_params(axis='x', which='major', length=4, width=2, labelsize=10)
                    else:
                        ax.set_xticklabels([])
                    
                    ax.tick_params(axis='y', which='major', length=4, width=2, labelsize=10)
                
                del shap_values, window_data
                torch.cuda.empty_cache()
        
        cax = fig.add_subplot(outer_gs[0, 1])
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.set_ylabel('Equalized SHAP Value', fontsize=16, fontweight='bold', 
                        rotation=-90, labelpad=20)
        
        tick_locations = [-1, -0.5, 0, 0.5, 1]
        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels([f'{val:g}' for val in tick_locations])
        cbar.ax.tick_params(labelsize=12, length=8, width=2)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='svg')
        plt.close('all')
    def find_patient_windows(self, results, test_pts, location) -> Dict[str, Dict[str, List[Dict]]]:
        """Find windows for each patient, organized by location"""
        patient_location_windows = defaultdict(lambda: defaultdict(list))
        fold = self.model_configs[0].fold  # Use first model's fold for threshold
        thresh = results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        
        for fold_key, fold_data in results['fold_metrics'].items():
            if 'detailed_results' in fold_data:
                for patient, patient_data in fold_data['detailed_results'].items():
                    if patient not in test_pts:
                        continue
                        
                    has_loc = any(location in loc_key for loc_key in patient_data['locations'].keys())
                    if not has_loc:
                        continue
                    for loc_name in patient_data['locations'].keys():
                        if location in loc_name:
                            if 'lead_combinations' in patient_data['locations'][loc_name]:
                                for lead_combo in patient_data['locations'][loc_name]['lead_combinations'].keys():
                                    label = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["true_label"]
                                    prediction = float(patient_data['locations'][loc_name]['lead_combinations'][lead_combo]['prediction'] >= thresh)
                                    correct = (prediction == label)
                                    
                                    if correct and label == 1:
                                        preds = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["windows"]["predictions"]
                                        confs = patient_data['locations'][loc_name]['lead_combinations'][lead_combo]["windows"]["confidences"]
                                        
                                        max_conf = 0
                                        for i in range(len(preds)):
                                            if confs[i] > max_conf:
                                                max_conf = confs[i]
                                                
                                            window_info = {
                                                'patient': patient,
                                                'leads': lead_combo.split("_"),
                                                'window_index': i,
                                                'confidence': confs[i],
                                                'prediction': preds[i],
                                                'true_label': label,
                                                'location': loc_name
                                            }
                                            patient_location_windows[patient][loc_name].append(window_info)
        
        for patient in patient_location_windows:
            for location in patient_location_windows[patient]:
                patient_location_windows[patient][location].sort(
                    key=lambda x: x['confidence'], 
                    reverse=True
                )
                
        return patient_location_windows
    def compute_shap_values(self, model_idx: int, window_info: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Compute SHAP values for a specific window and model"""
        patient = window_info['patient']
        leads = window_info['leads']
        window_start = window_info['window_index'] * self.datasets[model_idx].stride
        location = window_info['location']
        
        metadata = self.datasets[model_idx]._get_patient_metadata(patient)
        lead_indices = [metadata['locations'][location]['leads'].index(lead) for lead in leads]
        mmap_data = self.datasets[model_idx].get_mmap(patient)
        data = mmap_data[location]
        window_data = data[lead_indices, window_start:window_start+self.datasets[model_idx].window]
        
        window_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
        
        try:
            explainer = shap.GradientExplainer(self.models[model_idx], self.background_data[model_idx])
            shap_values = explainer.shap_values(window_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if torch.is_tensor(shap_values):
                shap_values = shap_values.cpu().numpy()
                
            return window_data, shap_values
            
        except Exception as e:
            print(f"Error computing SHAP values: {str(e)}")
            return None
        
    def normalize_shap_magnitudes(self, shap_values_list):
        """Normalize SHAP values from both models to comparable scales"""
        max_magnitude = max(np.max(np.abs(shap)) for shap in shap_values_list)
        print(f"\nGlobal max SHAP magnitude: {max_magnitude:.2e}")
        
        normalized_shaps = [shap / max_magnitude for shap in shap_values_list]
        
        for i, shap in enumerate(normalized_shaps):
            print(f"Model {i} normalized SHAP range: {np.min(shap):.2e} to {np.max(shap):.2e}")
        
        return normalized_shaps
    def normalize_shap_values(self, shap_values):
        """Normalize SHAP values to [-1, 1] range"""
        abs_max = np.max(np.abs(shap_values))
        if abs_max > 0:
            return shap_values / abs_max
        return shap_values
    
    def compute_model_max_shap_per_region(self, model_idx: int, windows_by_region: dict, chosen_regions: list, top_percent=0.2):
        """Compute maximum absolute SHAP value for a model using top windows per region"""
        region_max_shaps = {}
        
        counter = 0
        for region, windows in windows_by_region.items():
            sorted_windows = sorted(windows, key=lambda x: x['confidence'], reverse=True)
            chosen_windows = [window for window in sorted_windows if window['location'] == chosen_regions[counter]]
            if len(chosen_windows) == 0:
                region_max_shaps[region] = 0
            else:
                n_top = min(100, len(chosen_windows)) #top 100 or all
                top_windows = chosen_windows[:n_top]
                
                print(f"\nComputing max SHAP for model {model_idx}, region {region}, specifically for {chosen_regions[counter]}, using top {n_top} windows...")
                
                max_abs_shap = 0
                for window_info in top_windows:
                    result = self.compute_shap_values(model_idx, window_info)
                    if result is not None:
                        _, shap_values = result
                        for lead_shap in shap_values:
                            if np.any(np.isnan(lead_shap)) or np.any(np.isinf(lead_shap)):
                                continue
                            smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
                            if np.any(np.isnan(smoothed_shap)) or np.any(np.isinf(smoothed_shap)):
                                continue
                            max_abs_shap = max(max_abs_shap, np.max(np.abs(smoothed_shap)))
                
                region_max_shaps[region] = max_abs_shap
                counter += 1
            
            print(f"Model {model_idx}, {region} max absolute SHAP value: {max_abs_shap:.2e}")
                
        return region_max_shaps
    
    def compute_model_max_shap(self, model_idx: int, windows_by_region: dict, sample_fraction=0.05):
        """Compute maximum absolute SHAP value for a model using sampled windows"""
        max_abs_shap = 0
        all_windows = []
        
        for region, windows in windows_by_region.items():
            all_windows.extend(windows)
        
        n_samples = max(1, int(len(all_windows) * sample_fraction))
        sampled_windows = random.sample(all_windows, n_samples)
        
        print(f"\nComputing max SHAP for model {model_idx} using {n_samples} windows...")
        
        for window_info in sampled_windows:
            result = self.compute_shap_values(model_idx, window_info)
            if result is not None:
                _, shap_values = result
                for lead_shap in shap_values:
                    if np.any(np.isnan(lead_shap)) or np.any(np.isinf(lead_shap)):
                        continue
                    smoothed_shap = gaussian_filter1d(lead_shap, sigma=100)
                    if np.any(np.isnan(smoothed_shap)) or np.any(np.isinf(smoothed_shap)):
                        continue
                    max_abs_shap = max(max_abs_shap, np.max(np.abs(smoothed_shap)))
        
        print(f"Model {model_idx} max absolute SHAP value: {max_abs_shap:.2e}")
        return max_abs_shap
            
    def analyze_model_performance(self):
        """Analyze agreement and performance differences between models"""
        print("Analyzing model performance and agreement...")
        
        stats = {
            'total_cases': 0,
            'agreement': 0,
            'robust_better': 0,
            'histogram_better': 0,
            'by_region': defaultdict(lambda: {
                'total': 0,
                'agreement': 0,
                'robust_better': 0,
                'histogram_better': 0
            })
        }
        
        robust_thresh = self.results[0]['fold_metrics'][self.model_configs[0].fold]['metrics']['lead_level']['optimal_threshold']
        hist_thresh = self.results[1]['fold_metrics'][self.model_configs[1].fold]['metrics']['lead_level']['optimal_threshold']
        
        for patient in self.test_pts[0]:  # Use first model's test set
            robust_results = self.results[0]['fold_metrics'][self.model_configs[0].fold]['detailed_results'].get(patient, {})
            hist_results = self.results[1]['fold_metrics'][self.model_configs[1].fold]['detailed_results'].get(patient, {})
            
            if not robust_results or not hist_results:
                continue
                
            for location in robust_results.get('locations', {}):
                if location not in hist_results.get('locations', {}):
                    continue
                    
                robust_loc = robust_results['locations'][location]
                hist_loc = hist_results['locations'][location]
                
                for lead_combo in robust_loc.get('lead_combinations', {}):
                    if lead_combo not in hist_loc.get('lead_combinations', {}):
                        continue
                        
                    robust_data = robust_loc['lead_combinations'][lead_combo]
                    hist_data = hist_loc['lead_combinations'][lead_combo]
                    
                    true_label = robust_data['true_label']
                    robust_pred = float(robust_data['prediction'] >= robust_thresh)
                    hist_pred = float(hist_data['prediction'] >= hist_thresh)
                    
                    stats['total_cases'] += 1
                    stats['by_region'][location]['total'] += 1
                    
                    if robust_pred == hist_pred:
                        stats['agreement'] += 1
                        stats['by_region'][location]['agreement'] += 1
                    
                    if robust_pred == true_label and hist_pred != true_label:
                        stats['robust_better'] += 1
                        stats['by_region'][location]['robust_better'] += 1
                    elif hist_pred == true_label and robust_pred != true_label:
                        stats['histogram_better'] += 1
                        stats['by_region'][location]['histogram_better'] += 1
        
        print("\nOverall Statistics:")
        print(f"Total cases analyzed: {stats['total_cases']}")
        print(f"Model agreement: {(stats['agreement']/stats['total_cases']*100):.1f}%")
        print(f"Robust model better: {(stats['robust_better']/stats['total_cases']*100):.1f}%")
        print(f"Histogram model better: {(stats['histogram_better']/stats['total_cases']*100):.1f}%")
        
        print("\nBy Region Statistics:")
        for region, region_stats in stats['by_region'].items():
            if region_stats['total'] > 0:
                print(f"\n{region}:")
                print(f"Total cases: {region_stats['total']}")
                print(f"Agreement: {(region_stats['agreement']/region_stats['total']*100):.1f}%")
                print(f"Robust better: {(region_stats['robust_better']/region_stats['total']*100):.1f}%")
                print(f"Histogram better: {(region_stats['histogram_better']/region_stats['total']*100):.1f}%")
        
        return stats
    
def main():
    model_configs = [
        ModelConfig(
            model_path="/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/model_states/fold_3_results_1218_perm4_alllocs_v2_model.pth",
            results_path="/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/results_json/complete_results_1218_perm4_alllocs_v2.json",
            fold="fold_3",
            normalize_method="robust"
        ),
        ModelConfig(
            model_path="/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/model_states/fold_3_results_108_perm4_alllocs_v2_hist_model.pth",
            results_path="/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/results_json/complete_results_108_perm4_alllocs_v2_hist.json",
            fold="fold_3",
            normalize_method="histogram"
        )
    ]
    
    target_regions = ["HIPPOCAMPUS", 'MIDDLETEMPORAL', 'SUPERIORFRONTAL']
    
    analyzer = MultiModelSHAPAnalyzer(
        model_configs=model_configs,
        device="cuda",
        target_regions=target_regions, 
        stats_only=False
    )
    
    base_dir = '/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/location_analysis_modelcompare'
    os.makedirs(base_dir, exist_ok=True)

    for i in range(10):
        analyzer.plot_model_disagreements(
        save_path=f'/home/sameer/all_location_restingSEEG_SOZ/shap_time_analysis/location_analysis_modelcompare/disagreement_v10_{i}.svg')
    
if __name__ == "__main__":
    main()