"""
Morphology-Based SHAP Analysis

This module performs SHAP analysis focusing on signal morphology characteristics
and their contribution to SOZ classification.

Main Functions:
    analyze_morphology_features: Extract and analyze morphological signal features
    plot_morphology_shap: Visualize morphology-specific feature importance
    compare_morphology_patterns: Compare patterns across different morphologies
    extract_waveform_features: Extract time-domain morphological features
"""

from collections import defaultdict
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import SymLogNorm
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from seeg_classify_all import * 
from multi_scale_ori import *
class MorphologySHAPAnalyzer:
    def __init__(self, model, dataset, dataloader, train_pts, results_path: str,
                 device='cuda', conf_threshold=0.6, sample_rate=500):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.dataloader = dataloader
        self.train_pts = train_pts
        self.conf_threshold = conf_threshold
        self.sample_rate = sample_rate
        self.window_size = 250  # Size for morphology snippets
        self.model.eval()
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
            
        self.background_data = self.prepare_background_data()
        os.makedirs('morphology_analysis', exist_ok=True)
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
    def extract_morphology_windows(self, signal: np.ndarray, shap_values: np.ndarray) -> List[Dict]:
        morphology_windows = []
        half_window = self.window_size // 2
        
        shap_values = shap_values.flatten()
        positive_mask = shap_values > 0
        shap_values[~positive_mask] = 0
        
        peak_indices, peak_properties = find_peaks(
            shap_values,
            height=np.percentile(shap_values[positive_mask], 98),  # More selective
            distance=self.window_size,
            prominence=np.percentile(shap_values[positive_mask], 90)  # Add prominence threshold
        )
        
        for peak_idx, prominence in zip(peak_indices, peak_properties['prominences']):
            if peak_idx - half_window >= 0 and peak_idx + half_window < len(signal):
                window = {
                    'signal': signal[peak_idx-half_window:peak_idx+half_window],
                    'shap': shap_values[peak_idx-half_window:peak_idx+half_window],
                    'magnitude': shap_values[peak_idx],
                    'prominence': prominence
                }
                morphology_windows.append(window)
        
        return sorted(morphology_windows, key=lambda x: x['prominence'], reverse=True)[:50]  # Take top 50 by prominence
    
    def cluster_morphologies(self, windows: List[Dict], n_clusters=5):
        """
        Cluster morphology patterns using KMeans
        Args:
            windows: List of dictionaries containing signal data
            n_clusters: Number of clusters to create
        """
        
        print(f"Windows type: {type(windows)}")
        if len(windows) > 0:
            print(f"First window type: {type(windows[0])}")
            print(f"First window keys: {windows[0].keys() if isinstance(windows[0], dict) else 'Not a dict'}")
        
        if len(windows) < n_clusters:
            print(f"Warning: Only {len(windows)} windows for {n_clusters} clusters")
            n_clusters = max(1, len(windows))
        
        if len(windows) == 0:
            return {0: []}  # Return empty cluster
        
        try:
            signals = np.array([w['signal'] for w in windows])
            if len(signals.shape) > 2:
                signals = signals.reshape(signals.shape[0], -1)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(signals)
            
            clustered_windows = defaultdict(list)
            for idx, cluster in enumerate(clusters):
                clustered_windows[cluster].append(windows[idx])
            
            return clustered_windows
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            print(f"Windows shape: {np.array(windows).shape if isinstance(windows, (list, np.ndarray)) else 'Not array-like'}")
            return {0: windows}  # Return all windows in single cluster
  
    def collect_morphology_data(self) -> Dict[str, List]:
        """Collect morphology windows for each location"""
        location_morphologies = defaultdict(list)  # No longer using positive/negative dict
        fold = 'fold_5'
        thresh = self.results['fold_metrics'][fold]['metrics']['lead_level']['optimal_threshold']
        
        for fold_key, fold_data in self.results['fold_metrics'].items():
            if 'detailed_results' not in fold_data or fold_key != fold:
                continue
                
            for patient, patient_data in fold_data['detailed_results'].items():
                for location, loc_data in patient_data['locations'].items():
                    if 'lead_combinations' not in loc_data:
                        continue
                    for lead_combo, combo_data in loc_data['lead_combinations'].items():                        
                        label = patient_data['locations'][location]['lead_combinations'][lead_combo]["true_label"]
                        prediction = float(combo_data['prediction'] >= thresh)
                        correct = (prediction == label)
                        
                        if correct and label == 1:
                            preds = combo_data["windows"]["predictions"]
                            confs = combo_data["windows"]["confidences"]
                            max_conf = 0
                            index_to_use = 0
                            
                            for i in range(len(preds)):
                                if confs[i] > max_conf:
                                    max_conf = confs[i]
                                    index_to_use = i
                
                            try:
                                window_info = {
                                    'patient': patient,
                                    'leads': lead_combo.split("_"),
                                    'location': location,
                                    'window_index': index_to_use
                                }
                                
                                window_data, shap_values = self.compute_shap_values(window_info)
                                
                                for lead_idx in range(len(window_data)):
                                    morphology_windows = self.extract_morphology_windows(
                                        window_data[lead_idx],
                                        shap_values[lead_idx]
                                    )
                                    location_morphologies[location].extend(morphology_windows)
                                    
                            except Exception as e:
                                print(f"Error processing {location} in {patient}: {str(e)}")
                                continue
        
        return location_morphologies
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
        
        explainer = shap.GradientExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(window_tensor)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.squeeze(shap_values)  # Remove singleton dimensions
        shap_values = shap_values[:, :15000]  # Keep only relevant time points
            
        return window_data, shap_values
    def find_optimal_clusters(self, windows: List[Dict], max_clusters=10):
        """
        Use elbow method to find optimal number of clusters
        """
        if len(windows) < 3:  # Handle very small number of windows
            return 1
            
        signals = np.array([w['signal'] for w in windows])
        distortions = []
        
        K = range(1, min(max_clusters + 1, len(signals)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(signals)
            distortions.append(kmeans.inertia_)
        
        diffs = np.diff(distortions)
        elbow = np.argmin(diffs) + 1  # Add 1 because diff reduces length by 1
        
        optimal_k = min(max(2, elbow), max_clusters)
        print(f"Optimal clusters found: {optimal_k} (from elbow method)")
        return optimal_k
    
    def plot_morphology_comparison(self, location_set: List[str] = None, save_path: str = None):
        """
        Plot morphology patterns for locations.
        Args:
            location_set: Optional list of locations to plot. If None, plots all available locations.
            save_path: Path to save the figure.
        """
        location_morphologies = self.collect_morphology_data()
        
        if location_set is None:
            location_set = sorted(list(location_morphologies.keys()))
        print(f"Plotting locations: {location_set}")
        
        location_clusters = {}
        max_clusters = 0
        global_ymin = float('inf')
        global_ymax = float('-inf')
        
        for loc in location_set:
            if loc in location_morphologies:
                n_clusters = self.find_optimal_clusters(location_morphologies[loc])
                location_clusters[loc] = n_clusters
                max_clusters = max(max_clusters, n_clusters)
                
                windows = location_morphologies[loc]
                if windows:
                    signals = np.array([w['signal'] for w in windows])
                    global_ymin = min(global_ymin, np.min(signals))
                    global_ymax = max(global_ymax, np.max(signals))
        
        y_range = global_ymax - global_ymin
        global_ymin -= 0.1 * y_range
        global_ymax += 0.1 * y_range
        
        n_cols = len(location_set)
        fig = plt.figure(figsize=(6*n_cols, 4*max_clusters))  # Adjusted figsize for column layout
        
        plt.rcParams.update({
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.titleweight': 'bold',
            'font.family': 'sans-serif'
        })
        
        gs = GridSpec(max_clusters, n_cols, figure=fig)
        gs.update(hspace=0.3, wspace=0.4)  # Adjusted spacing
        
        for j, loc in enumerate(location_set):
            if loc not in location_morphologies:
                print(f"Warning: No data for location {loc}")
                continue
                
            windows = location_morphologies[loc]
            n_clusters = location_clusters[loc]
            print(f"Processing {loc} with {len(windows)} windows using {n_clusters} clusters")
            clustered = self.cluster_morphologies(windows, n_clusters)
            
            for i in range(max_clusters):
                ax = fig.add_subplot(gs[i, j])
                
                if i < n_clusters and i in clustered:
                    cluster_windows = clustered[i]
                    
                    if len(cluster_windows) > 0:
                        signals = np.array([w['signal'] for w in cluster_windows])
                        prominences = np.array([w['prominence'] for w in cluster_windows])
                        time_points = np.arange(len(signals[0])) / self.sample_rate
                        
                        for sig, prom in zip(signals, prominences):
                            alpha = min(1.0, prom / max(prominences) * 0.2)
                            ax.plot(time_points, sig, 'k-', alpha=alpha, linewidth=0.5)
                        
                        best_idx = np.argmax(prominences)
                        ax.plot(time_points, signals[best_idx], 'r-', linewidth=4, 
                            label='Representative', zorder=10)
                
                ax.grid(False)
                ax.set_ylim(global_ymin, global_ymax)  # Set consistent y-limits
                
                if i == max_clusters - 1:
                    ax.set_xlabel('Time (seconds)', fontsize=16, weight='bold')
                else:
                    ax.set_xticklabels([])
                
                if j == 0:
                    ax.set_ylabel(f'Pattern {i+1}', fontsize=16, weight='bold')
                else:
                    ax.set_yticklabels([])
                
                ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontweight('bold')
                
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color('black')
                
                if i == 0:
                    loc_display = loc.replace('CTX-', '').replace('-', ' ').title()
                    ax.set_title(loc_display, fontsize=18, pad=20, weight='bold')
                    
                    if j == 0:
                        ax.legend(loc='upper right', fontsize=14, frameon=False)
        
        plt.suptitle('Location-Specific Neural Pattern Morphologies', 
                    fontsize=24, y=0.95, weight='bold')
        
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5,
                    facecolor='white', edgecolor='none')
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
    analyzer = MorphologySHAPAnalyzer(model, dataset, dataloader, train_pts, results_path, device)
    
    location_set = [
        "CTX-LH-INFERIORTEMPORAL",
        "CTX-RH-LATERALORBITOFRONTAL",
        "CTX-LH-SUPERIORTEMPORAL",
    ]
    save_path = '/home/sameer/all_location_restingSEEG_SOZ/morphology_analysis/morphology_comparison_test_v4.png'
    analyzer.plot_morphology_comparison(save_path)
if __name__ == "__main__":
    main()