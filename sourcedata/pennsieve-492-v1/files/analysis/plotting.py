"""
Visualization and Statistical Analysis for SEEG Classification Results

This module provides comprehensive visualization and statistical analysis tools
for evaluating seizure onset zone (SOZ) classification models.

Key Features:
    - ROC curves and precision-recall curves with confidence intervals
    - Confusion matrices and performance metrics visualization
    - Location-specific and patient-specific analysis
    - Statistical significance testing
    - Publication-quality figure generation

Main Functions:
    plot_roc_curves: Generate ROC curves with confidence intervals
    plot_confusion_matrix: Visualize classification confusion matrices
    plot_location_analysis: Location-specific performance breakdown
    compute_statistics: Calculate comprehensive performance metrics
"""

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve

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
def load_results(file_path: str) -> Dict:
    """Load and process the results JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
def extract_metrics(fold_data: Dict, is_nested: bool = False, level='location') -> Dict:
    """Extract metrics handling both original and nested data structures.
        Level = "location" or "lead"
    """
    if is_nested:
        level_locorlead = level+'_level'
        metrics = fold_data['metrics'][level_locorlead] # CHANGE IF NEEDED
    else:
        metrics = fold_data['metrics']
    
    return {
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'accuracy': metrics['accuracy'],
        'youdens_index': metrics['youdens_index']
    }
    
def create_metrics_plot(df: pd.DataFrame, 
                       youden_p: float,
                       acc_p: float, 
                       title: str,
                       save_path: Optional[str] = None,
                       figsize: tuple = (10, 7)) -> None:
    """Create a standardized violin plot with consistent styling and separate y-axis for Youden."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 2]}, figsize=figsize)
    
    colors = {
        'Sensitivity': '#2C85B2',    # Steel blue
        'Specificity': '#569DAA',    # Lighter blue
        'Accuracy': '#87CBB9',       # Sea green
        'Accuracy Random Label': '#D3E5E1',  # Light sage
        "Youden": '#B9EDDD',         # Light turquoise
        "Youden Random Label": '#E5E1DA'  # Light gray
    }
    
    left_metrics = ['Sensitivity', 'Specificity', 'Accuracy', 'Accuracy Random Label']
    right_metrics = ['Youden', 'Youden Random Label']
    
    df_left = df[df['Metric'].isin(left_metrics)].copy()
    df_right = df[df['Metric'].isin(right_metrics)].copy()
    
    vl1 = sns.violinplot(data=df_left, 
                        x='Metric', 
                        y='Value',
                        order=left_metrics,
                        palette=[colors[m] for m in left_metrics],
                        inner='box',
                        linewidth=2,
                        saturation=0.7,
                        cut=0.1,
                        ax=ax1)
    
    vl2 = sns.violinplot(data=df_right, 
                        x='Metric', 
                        y='Value',
                        order=right_metrics,
                        palette=[colors[m] for m in right_metrics],
                        inner='box',
                        linewidth=2,
                        saturation=0.7,
                        cut=0.1,
                        ax=ax2)
    
    sns.stripplot(data=df_left, 
                 x='Metric', 
                 y='Value',
                 order=left_metrics,
                 color='black',
                 size=4,
                 alpha=0.3,
                 jitter=0.15,
                 ax=ax1)
    
    sns.stripplot(data=df_right, 
                 x='Metric', 
                 y='Value',
                 order=right_metrics,
                 color='black',
                 size=4,
                 alpha=0.3,
                 jitter=0.15,
                 ax=ax2)
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
    
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            if spine.get_visible():
                spine.set_linewidth(3)
    
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)  # Set limit to 0.7 for Youden
    
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    
    for ax in [ax1, ax2]:
        ax.tick_params(width=3, length=7, labelsize=16)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
    
    ax1.set_xticklabels(left_metrics, 
                      rotation=30,
                      ha='right',
                      fontsize=16,
                      fontweight='bold')
    ax2.set_xticklabels(right_metrics,
                      rotation=30,
                      ha='right',
                      fontsize=16,
                      fontweight='bold')
    
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    ax1.set_ylabel('Performance Score', fontsize=16, fontweight='bold', labelpad=15)
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=2)
    ax2.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3, linewidth=2)
    
    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)
    
    if acc_p < 0.05:
        x1, x2 = 2, 3  # Positions for Accuracy comparison
        y = df_left['Value'].max() + 0.05
        ax1.plot([x1, x2], [y, y], '-k', linewidth=3)
        ax1.plot([x1, x1], [y-0.01, y], '-k', linewidth=3)
        ax1.plot([x2, x2], [y-0.01, y], '-k', linewidth=3)
        ax1.text((x1 + x2) / 2, y + 0.01, '*',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=24,
                fontweight='bold')
    
    if youden_p < 0.05:
        x1, x2 = 0, 1  # Positions for Youden comparison
        y = df_left['Value'].max() + 0.05
        ax2.plot([x1, x2], [y, y], '-k', linewidth=3)
        ax2.plot([x1, x1], [y-0.01, y], '-k', linewidth=3)
        ax2.plot([x2, x2], [y-0.01, y], '-k', linewidth=3)
        ax2.text((x1 + x2) / 2, y + 0.01, '*',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=24,
                fontweight='bold')
    
    plt.subplots_adjust(wspace=0.4)  # Adjust spacing between subplots
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        plt.close()
    
    return fig, (ax1, ax2)
    
def process_results(results: Dict, 
                   random_results: Dict, 
                   is_nested: bool = False, 
                   level='location') -> tuple:
    """Process results and return DataFrame and p-values with confidence intervals."""
    metrics_data = []
    
    print(f"\nDebug - Processing main results for level {level}:")
    for fold_key, fold_data in results['fold_metrics'].items():
        metrics = extract_metrics(fold_data, is_nested, level)
        print(f"\nFold {fold_key}:")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Youden: {metrics['youdens_index']}")
        metrics_data.extend([
            {'Metric': 'Sensitivity', 'Value': metrics['sensitivity']},
            {'Metric': 'Specificity', 'Value': metrics['specificity']},
            {'Metric': 'Accuracy', 'Value': metrics['accuracy']},
            {'Metric': "Youden", 'Value': metrics['youdens_index']}
        ])
    
    print("\nDebug - Processing random results:")
    random_metrics_data = []
    for fold_key, fold_data in random_results['fold_metrics'].items():
        metrics = extract_metrics(fold_data, is_nested, level)
        print(f"\nRandom Fold {fold_key}:")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Youden: {metrics['youdens_index']}")
        random_metrics_data.extend([
            {'Metric': "Accuracy Random Label", 'Value': metrics['accuracy']},
            {'Metric': "Youden Random Label", 'Value': metrics['youdens_index']}
        ])
    
    df = pd.DataFrame(metrics_data)
    random_df = pd.DataFrame(random_metrics_data)
    
    youden_real = df[df['Metric'] == 'Youden']['Value'].values
    youden_random = random_df[random_df['Metric'] == 'Youden Random Label']['Value'].values
    acc_real = df[df['Metric'] == 'Accuracy']['Value'].values
    acc_random = random_df[random_df['Metric'] == 'Accuracy Random Label']['Value'].values
    
    print("\nNormality Tests (Shapiro-Wilk):")
    normality_results = {}
    
    for data_name, data in [("Youden Real", youden_real), 
                           ("Youden Random", youden_random),
                           ("Accuracy Real", acc_real),
                           ("Accuracy Random", acc_random)]:
        if len(data) > 3:
            stat, p_val = stats.shapiro(data)
            normality_results[data_name] = {"statistic": stat, "p_value": p_val, "normal": p_val > 0.05}
            print(f"{data_name}: statistic={stat:.4f}, p={p_val:.4f}, normal={p_val > 0.05}")
        else:
            normality_results[data_name] = {"statistic": None, "p_value": None, "normal": False}
            print(f"{data_name}: Sample size too small for normality test (n={len(data)})")

    def get_ci(data):
        ci_lower, ci_upper = stats.bootstrap((data,), 
                                           np.mean, 
                                           n_resamples=10000,
                                           confidence_level=0.95).confidence_interval
        return ci_lower, ci_upper
    
    metrics_ci = {}
    for metric in ['Sensitivity', 'Specificity', 'Accuracy', 'Youden']:
        data = df[df['Metric'] == metric]['Value'].values
        ci_lower, ci_upper = get_ci(data)
        metrics_ci[metric] = {
            'mean': np.mean(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    random_metrics_ci = {}
    for metric in ['Accuracy Random Label', 'Youden Random Label']:
        data = random_df[random_df['Metric'] == metric]['Value'].values
        ci_lower, ci_upper = get_ci(data)
        random_metrics_ci[metric] = {
            'mean': np.mean(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    print("\nDetailed Statistics with 95% Confidence Intervals:")
    print("\nMain Metrics:")
    for metric, stats_dict in metrics_ci.items():
        print(f"\n{metric}:")
        print(f"Mean: {stats_dict['mean']:.3f}")
        print(f"95% CI: [{stats_dict['ci_lower']:.3f}, {stats_dict['ci_upper']:.3f}]")
    
    print("\nRandom Label Metrics:")
    for metric, stats_dict in random_metrics_ci.items():
        print(f"\n{metric}:")
        print(f"Mean: {stats_dict['mean']:.3f}")
        print(f"95% CI: [{stats_dict['ci_lower']:.3f}, {stats_dict['ci_upper']:.3f}]")
    
    print("\nDebug - Values being compared:")
    print("Youden Real:", youden_real)
    print("Youden Random:", youden_random)
    print("Accuracy Real:", acc_real)
    print("Accuracy Random:", acc_random)
    
    youden_stat, youden_p = stats.mannwhitneyu(youden_real, youden_random, 
                                              alternative='greater')
    acc_stat, acc_p = stats.mannwhitneyu(acc_real, acc_random,
                                        alternative='greater')
    
    print("\nDebug - Statistical test results:")
    print(f"Youden - stat: {youden_stat}, p: {youden_p}")
    print(f"Accuracy - stat: {acc_stat}, p: {acc_p}")
    
    df_combined = pd.concat([df, random_df])
    
    return df_combined, youden_p, youden_stat, acc_p, acc_stat
def plot_comparison_metrics(overall_results_file: str, 
                          overall_random_file: str,
                          hippo_results_file: str,
                          hippo_random_file: str,
                          save_dir: str, 
                          level = "location") -> tuple:
    """Generate both overall and hippocampus-specific plots."""
    
    overall_results = load_results(overall_results_file)
    overall_random = load_results(overall_random_file)
    hippo_results = load_results(hippo_results_file)
    hippo_random = load_results(hippo_random_file)
    
    overall_df, overall_youden_p, overall_youden_stat, overall_acc_p, overall_acc_stat = process_results(
        overall_results, overall_random, is_nested=True, level=level)
    
    print("\nOverall Model Analysis:")
    print(f"Mann-Whitney U test (Youden): statistic={overall_youden_stat}, p-value={overall_youden_p:.4f}")
    print(f"Mann-Whitney U test (Accuracy): statistic={overall_acc_stat}, p-value={overall_acc_p:.4f}")
    
    for name, df in [("Overall", overall_df)]:
        print(f"\n{name} Model Summary Statistics:")
        for metric in df['Metric'].unique():
            values = df[df['Metric'] == metric]['Value']
            print(f"\n{metric}:")
            print(f"Mean ± SD: {values.mean():.3f} ± {values.std():.3f}")
            print(f"Range: [{values.min():.3f}, {values.max():.3f}]")
    
    create_metrics_plot(overall_df, overall_youden_p, overall_acc_p,
                       f"Overall Model Performance {level}-wise",
                       f"{save_dir}/overall_metrics_{level}_v4.svg")
    
    hippo_df = None
    return overall_df, hippo_df
def extract_hippocampus_metrics(results: Dict, hip_to_all = False) -> List[Dict]:
    """Extract metrics for hippocampus locations from the full model results."""
    hippo_metrics = []
    per_patient_preds = {}
    for fold_key, fold_data in results['fold_metrics'].items():
        print(f"\nProcessing fold {fold_key}")
        all_predictions = []
        all_labels = []
        if 'detailed_results' in fold_data:
            for patient, patient_data in fold_data['detailed_results'].items():
                for location in patient_data['locations'].keys():
                    if not hip_to_all:
                        if 'HIPPOCAMPUS' in location.upper():
                            
                            if 'lead_combinations' in patient_data['locations'][location]:
                                for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                    all_predictions.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'])
                                    all_labels.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label'])
                    else:
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                all_predictions.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'])
                                all_labels.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label'])
        if all_predictions:  # Only process if we have data
            predictions = np.array(all_predictions)
            true_labels = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = thresholds[optimal_idx]
            
            binary_preds = (predictions >= optimal_threshold).astype(float)
            
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            acc = (tp + tn) / len(true_labels)
            youden = sens + spec - 1
            hippo_metrics.append({
                'Fold': fold_key,
                'Location': location,
                'Sensitivity': sens,
                'Specificity': spec,
                'Accuracy': acc,
                "Youden's Index": youden
            })
    
    print(f"\nFound {len(hippo_metrics)} hippocampus metrics across all folds")
    return hippo_metrics
def extract_hippocampus_metrics_old(results: Dict) -> List[Dict]:
    """Extract metrics for hippocampus locations from the full model results."""
    hippo_metrics = []
    
    for fold_key, fold_data in results['fold_metrics'].items():
        print(f"\nProcessing fold {fold_key}")
      
        all_predictions = []
        all_labels = []
        if 'detailed_patient_results' in fold_data:
            print("Found detailed results")
            for patient in fold_data['detailed_patient_results'].keys():
                for lead_combo in fold_data['detailed_patient_results'][patient].keys():
                    all_predictions.append(fold_data['detailed_patient_results'][patient][lead_combo]['weighted_prediction'])
                    all_labels.append(fold_data['detailed_patient_results'][patient][lead_combo]['true_label'])
        if all_predictions:  # Only process if we have data
            predictions = np.array(all_predictions)
            true_labels = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = thresholds[optimal_idx]
            
            binary_preds = (predictions >= optimal_threshold).astype(float)
            
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            acc = (tp + tn) / len(true_labels)
            youden = sens + spec - 1
            hippo_metrics.append({
                'Fold': fold_key,
                'Location': "HIPPOCAMPUS",
                'Sensitivity': sens,
                'Specificity': spec,
                'Accuracy': acc,
                "Youden's Index": youden
            })
    
    print(f"\nFound {len(hippo_metrics)} hippocampus metrics across all folds")
    return hippo_metrics
def extract_hippocampus_accuracy_old(results: Dict) -> List[Dict]:
    """Extract accuracy for hippocampus locations from the hippocampus model results."""
    hippo_metrics = []
    with open('/home/sameer/pt_master_dict.json', 'r') as f:
        pt_master_dict = json.load(f)
    patient_hippo_data = {}
    
    for fold_key, fold_data in results['fold_metrics'].items():
        for patient, patient_data in fold_data['detailed_patient_results'].items():
            if patient not in patient_hippo_data:
                patient_hippo_data[patient] = {
                    'left': {'correct': 0, 'total': 0},
                    'right': {'correct': 0, 'total': 0},
                }
            
            for lead_combo, lead_data in patient_data.items():
                lead = lead_data['leads'][0]
                if lead in pt_master_dict[patient]['r_leads']:
                    patient_hippo_data[patient]['right']['correct'] += int(lead_data['correct'])
                    patient_hippo_data[patient]['right']['total'] += 1
                elif lead in pt_master_dict[patient]['l_leads']:
                    patient_hippo_data[patient]['left']['correct'] += int(lead_data['correct'])
                    patient_hippo_data[patient]['left']['total'] += 1
            
            if patient_hippo_data[patient]['right']['total'] > 0:
                hippo_metrics.append({
                    'Fold': fold_key,
                    'Accuracy': patient_hippo_data[patient]['right']['correct'] / patient_hippo_data[patient]['right']['total']
                })
            if patient_hippo_data[patient]['left']['total'] > 0:
                hippo_metrics.append({
                    'Fold': fold_key,
                    'Accuracy': patient_hippo_data[patient]['left']['correct'] / patient_hippo_data[patient]['left']['total'] 
                })
    
    print(f"\nFound {len(hippo_metrics)} hippocampus metrics across all folds for the hippocampus model")
    return hippo_metrics
def extract_hippocampus_accuracy(results: Dict) -> List[Dict]:
    """Extract metrics for each hippocampus from the full model results."""
    hippo_metrics = []
    accuracies = []
    per_patient_preds = {}
    for fold_key, fold_data in results['fold_metrics'].items():
        all_predictions = []
        all_labels = []
        if 'detailed_results' in fold_data:
            for patient, patient_data in fold_data['detailed_results'].items():
                for location in patient_data['locations'].keys():
                    if 'HIPPOCAMPUS' in location.upper():
                        
                        if 'lead_combinations' in patient_data['locations'][location]:
                            
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                all_predictions.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'])
                                all_labels.append(patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label'])
        
        if all_predictions:  # Only process if we have data
            predictions = np.array(all_predictions)
            true_labels = np.array(all_labels)
            
            fpr, tpr, thresholds = roc_curve(true_labels, predictions)
            youdens_j = tpr - fpr
            optimal_idx = np.argmax(youdens_j)
            optimal_threshold = thresholds[optimal_idx]
            
            binary_preds = (predictions >= optimal_threshold).astype(float)
            
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            acc = (tp + tn) / len(true_labels)
            youden = sens + spec - 1
        if 'detailed_results' in fold_data and optimal_threshold:
            for patient, patient_data in fold_data['detailed_results'].items():
                for location in patient_data['locations'].keys():
                    if 'HIPPOCAMPUS' in location.upper():
                        num_correct = 0
                        num_total = 0
                        
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                pred = 0
                                if patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction'] >= optimal_threshold:
                                    pred = 1
                                if pred == patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label']:
                                    num_correct += 1
                                num_total += 1
                        
                        accuracy = num_correct / num_total
                        hippo_metrics.append({
                            'Fold': fold_key,
                            'Accuracy': accuracy
                        })
    
    print(f"\nFound {len(hippo_metrics)} hippocampus metrics across all folds")
    return hippo_metrics
def plot_accuracy_comparison(df_plot: pd.DataFrame, p_value: float, save_path: Optional[str] = None):
    """Create a bar plot comparison for hippocampus accuracies."""
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    
    stats_dict = {}
    for model in ['Full Model', 'Hippo Model']:
        model_data = df_plot[df_plot['Model'] == model]['Value']
        stats_dict[model] = {
            'mean': model_data.mean(),
            'sem': model_data.std() / np.sqrt(len(model_data))
        }
    
    colors = {
        'Full Model': '#2C85B2',    # Steel blue
        'Hippo Model': '#B9EDDD'    # Light turquoise
    }
    
    x_pos = np.arange(len(stats_dict))
    bars = plt.bar(x_pos, 
                  [stats_dict[m]['mean'] for m in ['Full Model', 'Hippo Model']],
                  yerr=[stats_dict[m]['sem'] for m in ['Full Model', 'Hippo Model']],
                  color=[colors[m] for m in ['Full Model', 'Hippo Model']],
                  capsize=8,
                  width=0.4,  # Reduced width
                  edgecolor='black',  # Add black outline
                  linewidth=2,  # Outline thickness
                  error_kw={'linewidth': 3, 'capthick': 3, 'ecolor': 'black'})  # Thicker error bars
    
    for idx, model in enumerate(['Full Model', 'Hippo Model']):
        data = df_plot[df_plot['Model'] == model]['Value']
        x = np.random.normal(idx, 0.04, size=len(data))
        plt.plot(x, data, 'o', color='black', alpha=0.4, markersize=6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tick_params(axis='both', which='major', length=7, width=3)
    plt.xticks(x_pos, ['Full Model\n(Hippo Only)', 'Dedicated\nHippo Model'], 
               fontsize=16, fontweight='bold', ha='center')  # Removed rotation
    plt.yticks(fontsize=16, fontweight='bold')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.title("Hippocampus Classification Accuracy", 
             fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold', labelpad=15)
    
    if p_value < 0.05:
        y_max = df_plot['Value'].max()
        y_bar = y_max + 0.05
        plt.plot([0-0.2, 1+0.2], [y_bar, y_bar], '-k', linewidth=2)
        plt.plot([0-0.2, 0-0.2], [y_bar-0.01, y_bar], '-k', linewidth=2)
        plt.plot([1+0.2, 1+0.2], [y_bar-0.01, y_bar], '-k', linewidth=2)
        plt.text(0.5, y_bar + 0.01, '*', 
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=20)
        plt.ylim(0, y_bar + 0.1)
    
    plt.legend(bars, ['Full Model', 'Hippo Model'],
              title='', loc='upper right',
              fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
def compare_hippocampus_accuracy(full_model_file: str, 
                                  hippo_model_file: str,
                                  save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hippocampus accuracy between full and hippocampus-only models."""
    
    with open(full_model_file, 'r') as f:
        full_results = json.load(f)
    with open(hippo_model_file, 'r') as f:
        hippo_results = json.load(f)
    
    full_hippo_metrics = extract_hippocampus_accuracy(full_results)
    
    hippo_only_metrics = extract_hippocampus_accuracy_old(hippo_results)
    
    full_df = pd.DataFrame(full_hippo_metrics)
    hippo_df = pd.DataFrame(hippo_only_metrics)
    
    print("\nFull model DataFrame columns:", full_df.columns.tolist())
    print("Hippo model DataFrame columns:", hippo_df.columns.tolist())
    
    metrics = ['Accuracy']
    stat, p_val = stats.mannwhitneyu(full_df['Accuracy'], 
                                   hippo_df['Accuracy'],
                                   alternative='two-sided')
    plot_data = []
    for val in full_df['Accuracy']:
        plot_data.append({
            'Value': val,
            'Model': 'Full Model'
        })
    for val in hippo_df['Accuracy']:
        plot_data.append({
            'Value': val,
            'Model': 'Hippo Model'
        })
    
    df_plot = pd.DataFrame(plot_data)
    plot_accuracy_comparison(df_plot, p_val, save_path)
    
    return full_df, hippo_df
def compare_hippocampus_performance(full_model_file: str, 
                                  hippo_model_file: str, hip_to_all = False,
                                  save_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare hippocampus performance between full and hippocampus-only models."""
    
    with open(full_model_file, 'r') as f:
        full_results = json.load(f)
    with open(hippo_model_file, 'r') as f:
        hippo_results = json.load(f)
    
    full_hippo_metrics = extract_hippocampus_metrics(full_results, hip_to_all)
    
    hippo_only_metrics = extract_hippocampus_metrics_old(hippo_results)
    
    full_df = pd.DataFrame(full_hippo_metrics)
    hippo_df = pd.DataFrame(hippo_only_metrics)
    
    print("\nFull model DataFrame columns:", full_df.columns.tolist())
    print("Hippo model DataFrame columns:", hippo_df.columns.tolist())
    
    metrics = ['Sensitivity', 'Specificity', 'Accuracy', "Youden's Index"]
    if hip_to_all:
        model_name = "Full Model (All Locs)"
    else:
        model_name = "Full Model (Hipp Only)"
    plot_data = []
    for metric in metrics:
        for val in full_df[metric]:
            plot_data.append({
                'Metric': metric,
                'Value': val,
                'Model': model_name
            })
        for val in hippo_df[metric]:
            plot_data.append({
                'Metric': metric,
                'Value': val,
                'Model': 'Hippo Model'
            })
    print("\nPerformance Comparison:")
    print("\nMetric               Full Model (Hippo Only)     Dedicated Hippo Model")
    print("-" * 65)
    
    for metric in metrics:
        full_mean = full_df[metric].mean()
        full_std = full_df[metric].std()
        hippo_mean = hippo_df[metric].mean()
        hippo_std = hippo_df[metric].std()
        
        stat, p_val = stats.mannwhitneyu(full_df[metric], hippo_df[metric], 
                                       alternative='two-sided')
        
        print(f"{metric:20} {full_mean:.3f} ± {full_std:.3f}          {hippo_mean:.3f} ± {hippo_std:.3f}")
        print(f"{'':20} p-value = {p_val:.4f}")
    
    if save_path:
        plt.figure(figsize=(12, 6))
        
        colors = {
            model_name: '#2C85B2',    # Steel blue
            'Hippo Model': '#B9EDDD'    # Light turquoise
        }
        
        ax = plt.gca()
        df_plot = pd.DataFrame(plot_data)
        
        sns.violinplot(data=df_plot, 
                      x='Metric', 
                      y='Value',
                      hue='Model',
                      palette=colors,
                      inner='box',
                      linewidth=1,
                      saturation=0.7,
                      cut=0.1,
                      split=False)
        
        sns.stripplot(data=df_plot, 
                     x='Metric', 
                     y='Value',
                     hue='Model',
                     dodge=True,
                     size=6,
                     alpha=0.4,
                     color='black', 
                     legend=False)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        
        plt.ylim(0, 1.2)
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.tick_params(axis='both', which='major', length=7, width=3)
        plt.xticks(rotation=30, ha='right', fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        if hip_to_all:
            title = "Fold-wise Lead-wise Classifcation Metrics"
        else:
            title = "Fold-wise Hippocampal Lead-wise Classifcation Metrics"
        plt.title(f"{title}", fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('Performance Score', fontsize=16, fontweight='bold', labelpad=15)
        
        for idx, metric in enumerate(metrics):
            stat, p_val = stats.mannwhitneyu(
                full_df[metric], 
                hippo_df[metric],
                alternative='two-sided'
            )
            
            if p_val < 0.05:
                y = df_plot[df_plot['Metric'] == metric]['Value'].max() + 0.05
                plt.plot([idx-0.2, idx+0.2], [y, y], '-k', linewidth=2)
                plt.text(idx, y + 0.01, '*', 
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=20)
        
        plt.legend(title='', loc='upper right',
                  fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        plt.close()
    
    return full_df, hippo_df
def extract_location_accuracies(results: Dict) -> List[Dict]:
    """Extract accuracy metrics for each location from the model results."""
    location_metrics = {}  # Dictionary to store metrics for each location
    
    for fold_key, fold_data in results['fold_metrics'].items():
        if 'detailed_results' in fold_data:
            for patient, patient_data in fold_data['detailed_results'].items():
                for location in patient_data['locations'].keys():
                    if location in location_set:
                        if location not in location_metrics:
                            location_metrics[location] = {
                                'predictions': [],
                                'labels': [],
                                'accuracies': [],
                                'num_patients':0
                            }
                        
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                location_metrics[location]['predictions'].append(
                                    patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction']
                                )
                                location_metrics[location]['labels'].append(
                                    patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label']
                                )
                            location_metrics[location]['num_patients'] += 1
                        
    metrics_list = []
    for location, data in location_metrics.items():
        if data['predictions']:  # Only process if we have data
            predictions = np.array(data['predictions'])
            true_labels = np.array(data['labels'])
            
            if 1 in true_labels and 0 in true_labels:
                
                fpr, tpr, thresholds = roc_curve(true_labels, predictions)
                youdens_j = tpr - fpr
                optimal_idx = np.argmax(youdens_j)
                optimal_youden = np.max(youdens_j)
                optimal_threshold = thresholds[optimal_idx]
                
                binary_preds = (predictions >= optimal_threshold).astype(float)
            else:
                binary_preds = (predictions >= 0.5).astype(float)
            
            acc = np.mean(binary_preds == true_labels)
            
            metrics_list.append({
                'Location': location,
                'Accuracy': acc,
                'N_samples': len(predictions),
                'N_patients': data['num_patients']
            })
    
    metrics_list = sorted(metrics_list, key=lambda x: x['Accuracy'], reverse=True)
    print(f"\nAnalyzed accuracy for {len(metrics_list)} locations")
    return metrics_list
    
def create_ordered_metrics(metrics_list: List[Dict]) -> List[Dict]:
    """Reorder metrics to group L/R pairs and maintain alphabetical order."""
    location_pairs = {}
    for metric in metrics_list:
        loc_name = metric['Location']
        base_name = loc_name.replace('CTX-LH-', '').replace('CTX-RH-', '')\
                          .replace('LEFT ', '').replace('RIGHT ', '')\
                          .replace('L ', '').replace('R ', '')
        
        if base_name not in location_pairs:
            location_pairs[base_name] = {'L': None, 'R': None}
        
        if any(prefix in loc_name for prefix in ['CTX-LH-', 'LEFT ', 'L ']):
            location_pairs[base_name]['L'] = metric
        elif any(prefix in loc_name for prefix in ['CTX-RH-', 'RIGHT ', 'R ']):
            location_pairs[base_name]['R'] = metric
    
    ordered_metrics = []
    for base_name in sorted(location_pairs.keys()):
        pair = location_pairs[base_name]
        if pair['L']:
            ordered_metrics.append(pair['L'])
        if pair['R']:
            ordered_metrics.append(pair['R'])
    
    return ordered_metrics
def plot_location_accuracies(metrics_list: List[Dict], save_path: Optional[str] = None):
    """Create a bar plot comparing accuracies across locations."""
    plt.figure(figsize=(25, 8))
    ax = plt.gca()
    
    locations = [m['Location'] for m in metrics_list]
    accuracies = [m['Accuracy'] for m in metrics_list]
    n_samples = [m['N_samples'] for m in metrics_list]
    n_patients = [m['N_patients'] for m in metrics_list]
    error_bars = [np.sqrt((acc * (1-acc)) / n) for acc, n in zip(accuracies, n_samples) if n >= 5] # limit to n>=5 
    
    x_pos = np.arange(len(locations))
    bars = plt.bar(x_pos, accuracies,
                  yerr=error_bars,
                  capsize=8,
                  width=0.6,
                  edgecolor='black',
                  linewidth=2,
                  error_kw={'linewidth': 3, 'capthick': 3, 'ecolor': 'black'})
    
    significant_bars = []
    nonsignificant_bars = []
    for i, (acc, err) in enumerate(zip(accuracies, error_bars)):
        if err == 0:
            err = 1e-6
        z_score = (acc - 0.5) / err
        p_value = 1 - stats.norm.cdf(z_score)
        
        if p_value < 0.05:
            bars[i].set_facecolor('#2C85B2')  # Steel blue for significant
            significant_bars.append(bars[i])
        else:
            bars[i].set_facecolor('#B9EDDD')  # Light turquoise for non-significant
            nonsignificant_bars.append(bars[i])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tick_params(axis='both', which='major', length=7, width=3)
    
    plt.xlim(-0.5, len(locations)-0.5)
    
    formatted_locations = [" ".join([loc.replace('CTX-', '')
                            .replace('RH', 'R ')
                            .replace('LH', 'L ')
                            .replace('LEFT', 'L ')
                            .replace('RIGHT', 'R ')
                            .replace('-', ' '), f"(n={n})"]) for loc,n in zip(locations,n_patients)]
    
    plt.xticks(x_pos, 
               formatted_locations,
               fontsize=12, fontweight='bold', 
               ha='right',  # Align text to the right
               rotation=45)
    
    plt.yticks(fontsize=16, fontweight='bold')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.title("Classification Accuracy by Location", 
             fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold', labelpad=15)
    
    if significant_bars:
        sig_bar = significant_bars[0]
    else:
        sig_bar = plt.Rectangle((0,0),1,1, fc='#2C85B2', ec='black', linewidth=2)
    
    if nonsignificant_bars:
        nonsig_bar = nonsignificant_bars[0]
    else:
        nonsig_bar = plt.Rectangle((0,0),1,1, fc='#B9EDDD', ec='black', linewidth=2)
        
    plt.legend([sig_bar, nonsig_bar],
              ['Significantly Above Chance', 'Not Significant'],
              loc='upper right',
              fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
    
    return metrics_list
    
def extract_patient_metrics(results: Dict) -> List[Dict]:
    """Extract metrics aggregated at the patient level from model results."""
    patient_metrics = {}
    
    for fold_key, fold_data in results['fold_metrics'].items():
        if 'detailed_results' in fold_data:
            for patient, patient_data in fold_data['detailed_results'].items():
                if patient not in patient_metrics:
                    patient_metrics[patient] = {
                        'predictions': [],
                        'labels': [],
                        'total_leads': 0
                    }
                
                for location in patient_data['locations'].keys():
                    if 'lead_combinations' in patient_data['locations'][location]:
                        for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                            patient_metrics[patient]['predictions'].append(
                                patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction']
                            )
                            patient_metrics[patient]['labels'].append(
                                patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label']
                            )
                            patient_metrics[patient]['total_leads'] += 1
    
    metrics_list = []
    for patient, data in patient_metrics.items():
        if data['predictions']:  # Only process if we have data
            predictions = np.array(data['predictions'])
            true_labels = np.array(data['labels'])
            
            if 1 in true_labels and 0 in true_labels:  # Only calculate metrics if we have both positive and negative samples
                fpr, tpr, thresholds = roc_curve(true_labels, predictions)
                youdens_j = tpr - fpr
                optimal_idx = np.argmax(youdens_j)
                optimal_threshold = thresholds[optimal_idx]
                optimal_youden = youdens_j[optimal_idx]
                
                binary_preds = (predictions >= optimal_threshold).astype(float)
                
                tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                acc = (tp + tn) / len(true_labels)
                youden = sens + spec - 1
                
                metrics_list.append({
                    'Patient': patient,
                    'Accuracy': acc,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    "Youden's Index": youden,
                    'N_leads': data['total_leads']
                })
            else:
                print(f"there are no defineable youden's index for {patient}. defaulting...")
                binary_preds = (predictions >= 0.5).astype(float)
                
                acc = np.mean(binary_preds == true_labels)
                
                metrics_list.append({
                    'Patient': patient,
                    'Accuracy': acc,
                    'Sensitivity': 0,
                    'Specificity': 0,
                    "Youden's Index": 0,
                    'N_leads': data['total_leads']
                })
    
    print(f"\nCalculated metrics for {len(metrics_list)} patients")
    return metrics_list

def extract_location_metrics(results: Dict, ordering = "Youden", combined=False) -> List[Dict]:
    """Extract metrics for each location using binomial confidence intervals."""
    location_metrics = {}
    
    for fold_key, fold_data in results['fold_metrics'].items():
        if 'detailed_results' in fold_data:
            for patient, patient_data in fold_data['detailed_results'].items():
                for location in patient_data['locations'].keys():
                    if location in location_set:
                        if combined:
                            comb_location = location.replace('CTX-', '').replace('RH', '').replace('LH', '').replace('LEFT', '').replace('RIGHT', '').replace('-', '')
                        else:
                            comb_location = location
                            
                        if comb_location not in location_metrics:
                            location_metrics[comb_location] = {
                                'predictions': [],
                                'labels': [],
                                'num_patients': set()  # Use set to count unique patients
                            }
                        
                        if 'lead_combinations' in patient_data['locations'][location]:
                            for lead_combo in patient_data['locations'][location]['lead_combinations'].keys():
                                location_metrics[comb_location]['predictions'].append(
                                    patient_data['locations'][location]['lead_combinations'][lead_combo]['prediction']
                                )
                                location_metrics[comb_location]['labels'].append(
                                    patient_data['locations'][location]['lead_combinations'][lead_combo]['true_label']
                                )
                            location_metrics[comb_location]['num_patients'].add(patient)
    metrics_list = []
    for location, data in location_metrics.items():
        if len(data['num_patients']) >= 5:  # Only process locations with enough patients
            predictions = np.array(data['predictions'])
            labels = np.array(data['labels'])
            
            if 1 in labels and 0 in labels:
                fpr, tpr, thresholds = roc_curve(labels, predictions)
                youdens_j = tpr - fpr
                optimal_idx = np.argmax(youdens_j)
                optimal_threshold = thresholds[optimal_idx]
                binary_preds = (predictions >= optimal_threshold).astype(float)
                optimal_youden = youdens_j[optimal_idx]
            else:
                binary_preds = (predictions >= 0.5).astype(float)
                optimal_youden = 0
            
            n_correct = np.sum(binary_preds == labels)
            n_total = len(labels)
            accuracy = n_correct / n_total
            
            z = 1.96  # 95% confidence
            denominator = 1 + z**2/n_total
            centre_adjusted_probability = (accuracy + z*z/(2*n_total))/denominator
            adjusted_standard_error = z * np.sqrt(accuracy*(1-accuracy)/n_total + z*z/(4*n_total*n_total))/denominator
            adjusted_standard_error_y = z * np.sqrt(optimal_youden*(1-optimal_youden)/n_total + z*z/(4*n_total*n_total))/denominator
            ci_lower = max(0.0, centre_adjusted_probability - adjusted_standard_error)
            ci_upper = min(1.0, centre_adjusted_probability + adjusted_standard_error)
            ci_lower_y = max(0.0, optimal_youden - adjusted_standard_error_y)
            ci_upper_y = min(1.0, optimal_youden + adjusted_standard_error_y)
            
            metrics_list.append({
                'Location': location,
                'Accuracy': accuracy,
                'Accuracy_CI': [ci_lower, ci_upper],
                'Youden_CI':[ci_lower_y, ci_upper_y],
                'Youden': optimal_youden,
                'N_samples': n_total,
                'N_patients': len(data['num_patients'])
            })
    
    metrics_list = sorted(metrics_list, key=lambda x: x[ordering], reverse=True)
    print(f"\nAnalyzed metrics for {len(metrics_list)} locations")
    
    print("\nDetailed Location Statistics:")
    for m in metrics_list:
        print(f"\n{m['Location']}:")
        print(f"Accuracy: {m['Accuracy']:.3f} [95% CI: {m['Accuracy_CI'][0]:.3f}, {m['Accuracy_CI'][1]:.3f}]")
        print(f"Youden: {m['Youden']:.3f}")
        print(f"N patients: {m['N_patients']}, N samples: {m['N_samples']}")
    
    return metrics_list
def plot_location_metrics(metrics_list: List[Dict], metric = 'Youden', save_path: Optional[str] = None):
    """Create a bar plot comparing metrics across locations using bootstrapped CIs."""
    plt.figure(figsize=(25, 8))
    ax = plt.gca()
    
    filtered_metrics = [m for m in metrics_list if int(m['N_patients']) > 5]
    locations = [m['Location'] for m in filtered_metrics]
    values = [m[metric] for m in filtered_metrics]
    ci_data = [m[f'{metric}_CI'] for m in filtered_metrics]
    n_patients = [m['N_patients'] for m in filtered_metrics]
    
    yerr = np.array([(v - ci[0], ci[1] - v) for v, ci in zip(values, ci_data)]).T
    
    x_pos = np.arange(len(locations))
    bars = plt.bar(x_pos, values,
                  yerr=yerr,
                  capsize=8,
                  width=0.6,
                  edgecolor='black',
                  linewidth=2,
                  error_kw={'linewidth': 3, 'capthick': 3, 'ecolor': 'black'})
    
    if metric == 'Accuracy':
        significant_bars = []
        nonsignificant_bars = []
        n_tests = len(values)  # Number of locations being tested
        bonferroni_alpha = 0.05 / n_tests  # Corrected significance threshold
        
        for i, (value, n) in enumerate(zip(values, [m['N_samples'] for m in filtered_metrics])):
            se = np.sqrt((value * (1-value)) / n)
            z_score = (value - 0.5) / se
            p_value = 1 - stats.norm.cdf(z_score)
            
            if p_value < bonferroni_alpha:
                bars[i].set_facecolor('#2C85B2')  # Steel blue for significant
                significant_bars.append(bars[i])
            else:
                bars[i].set_facecolor('#B9EDDD')  # Light turquoise for non-significant
                nonsignificant_bars.append(bars[i])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tick_params(axis='both', which='major', length=7, width=3)
    plt.xlim(-0.5, len(locations)-0.5)
    
    formatted_locations = [" ".join([loc.replace('CTX-', '')
                            .replace('RH', 'R ')
                            .replace('LH', 'L ')
                            .replace('LEFT', 'L ')
                            .replace('RIGHT', 'R ')
                            .replace('-', ' '), f"(n={n})"]) for loc,n in zip(locations,n_patients)]
    
    plt.xticks(x_pos, 
               formatted_locations,
               fontsize=12, fontweight='bold', 
               ha='right',
               rotation=45)
    
    plt.yticks(fontsize=16, fontweight='bold')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.title(f"Classification {metric} by Anatomical Location", 
             fontsize=18, fontweight='bold', pad=20)
    
    ax.set_xlabel('')
    ax.set_ylabel(f'{metric}', fontsize=16, fontweight='bold', labelpad=15)
    
    if metric == 'Accuracy':
        if significant_bars:
            sig_bar = significant_bars[0]
        else:
            sig_bar = plt.Rectangle((0,0),1,1, fc='#2C85B2', ec='black', linewidth=2)
        
        if nonsignificant_bars:
            nonsig_bar = nonsignificant_bars[0]
        else:
            nonsig_bar = plt.Rectangle((0,0),1,1, fc='#B9EDDD', ec='black', linewidth=2)
        
        plt.legend([sig_bar, nonsig_bar],
                ['Significantly Above Chance', 'Not Significant'],
                loc='upper right',
                fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        plt.close()
    
    return metrics_list
        
def plot_patient_outcome_metrics(metrics_list: List[Dict], 
                               dict_outcomes: Dict,
                               metric: str = 'Accuracy',
                               binary_comparison: bool = False,  # New parameter
                               save_path: Optional[str] = None):
    """Create a bar plot comparing patient-level metrics between outcome categories.
    
    Args:
        binary_comparison: If True, compares Engel I/RNS Responder vs All Others.
    """
    outcome_metrics = []
    for m in metrics_list:
        outcome = dict_outcomes.get(m['Patient'], None)
        
        if pd.isna(outcome) or outcome is None:
            category = 'Other Outcomes' if binary_comparison else 'Unknown'
        else:
            outcome = str(outcome)
            if (outcome.startswith('I-') or outcome.startswith('1-') or 
                outcome.startswith('1a') or outcome.startswith('RNS')):
                category = 'Engel I/RNS Responder'
            else:
                category = 'Other Outcomes' if binary_comparison else (
                    'Engel II-IV/RNS Non-Resp.' if (outcome.startswith('Non') or 
                    outcome.startswith('II') or outcome.startswith('III') or 
                    outcome.startswith('IV') or outcome.startswith('2-') or outcome.startswith('3-')
                    or outcome.startswith('4-'))
                    else 'Unknown'
                )
        
        if metric == "Youden's Index" and m[metric] == 0 and m['Sensitivity'] == 0 and m['Specificity'] == 0:
            continue
        else:
            outcome_metrics.append({
                'Patient': m['Patient'],
                'Outcome': category,
                'Value': m[metric],
                'N_leads': m['N_leads']
            })
    
    df = pd.DataFrame(outcome_metrics)
    
    list_cat = ['Engel I/RNS Responder', 'Other Outcomes'] if binary_comparison else [
        'Engel I/RNS Responder', 'Engel II-IV/RNS Non-Resp.', 'Unknown'
    ]
    
    stats_dict = {}
    for outcome in list_cat:
        print(outcome)
        outcome_data = df[df['Outcome'] == outcome]['Value']
        ci_lower, ci_upper = stats.bootstrap((outcome_data,), 
                                   np.mean, 
                                   n_resamples=10000,
                                   confidence_level=0.95).confidence_interval
        stats_dict[outcome] = {
            'mean': outcome_data.mean(),
            'sem': (ci_upper - ci_lower) / 2
        }
        print(f"mean = {outcome_data.mean()} [{ci_lower}, {ci_upper}]")
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    
    colors = {
        'Engel I/RNS Responder': '#2C85B2',    # Steel blue
        'Other Outcomes': '#B9554B',            # Rust red
        'Engel II-IV/RNS Non-Resp.': '#B9554B', # Rust red
        'Unknown': '#E5E1DA'                    # Light grey
    }
    
    x_pos = np.arange(len(stats_dict))
    bars = plt.bar(x_pos, 
                  [stats_dict[m]['mean'] for m in list_cat],
                  yerr=[stats_dict[m]['sem'] for m in list_cat],
                  color=[colors[m] for m in list_cat],
                  capsize=8,
                  width=0.4,
                  edgecolor='black',
                  linewidth=2,
                  error_kw={'linewidth': 3, 'capthick': 3, 'ecolor': 'black'})
    
    for idx, outcome in enumerate(list_cat):
        data = df[df['Outcome'] == outcome]['Value']
        x = np.random.normal(idx, 0.04, size=len(data))
        plt.plot(x, data, 'o', color='black', alpha=0.4, markersize=6)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.tick_params(axis='both', which='major', length=7, width=3)
    
    outcome_counts = df.groupby('Outcome').size()
    plt.xticks(x_pos, 
               [f"{outcome}\n(n={outcome_counts.get(outcome, 0)})" for outcome in list_cat], 
               fontsize=16, fontweight='bold', rotation=15, ha='right')
    plt.yticks(fontsize=16, fontweight='bold')
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.title(f"Patient-Level {metric} by Outcome", 
             fontsize=18, fontweight='bold', pad=20)
    plt.ylabel(metric, fontsize=16, fontweight='bold', labelpad=15)
    
    if binary_comparison:
        data1 = df[df['Outcome'] == 'Engel I/RNS Responder']['Value']
        data2 = df[df['Outcome'] == 'Other Outcomes']['Value']
        
        if len(data1) > 0 and len(data2) > 0:
            stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            if p_val < 0.05:  # No correction needed for single comparison
                y_max = df['Value'].max()
                y_bar = y_max + 0.05
                
                plt.plot([0, 1], [y_bar, y_bar], '-k', linewidth=2)
                plt.text(0.5, y_bar + 0.005, '*', 
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=20,
                        fontweight='bold')
                
                plt.ylim(0, y_bar + 0.05)
    else:
        y_max = df['Value'].max()
        spacing = 0.025  # Spacing between significance bars
        pairs = [
            (0, 1, 'Engel I/RNS Responder', 'Engel II-IV/RNS Non-Resp.'),
            (1, 2, 'Engel II-IV/RNS Non-Resp.', 'Unknown'),
            (0, 2, 'Engel I/RNS Responder', 'Unknown')
        ]
        ntests = len(list(enumerate(pairs)))
        print(f"Number of tests = {ntests}")
        pvals = []
        for i, (idx1, idx2, group1, group2) in enumerate(pairs):
            data1 = df[df['Outcome'] == group1]['Value']
            data2 = df[df['Outcome'] == group2]['Value']
            
            if len(data1) > 0 and len(data2) > 0:  # Only test if both groups have data
                stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                pvals.append((idx1, idx2, group1, group2, p_val))
        
        pvals.sort(key=lambda x: x[4])
        for i in range(len(pvals)):
            idx1, idx2 = pvals[i][0], pvals[i][1]
            group1 = pvals[i][2]
            group2 = pvals[i][3]
            p_val = pvals[i][4]
            print(f"pair = {group1}, {group2}, uncorrect p = {p_val}")
            if p_val < (0.05 / (ntests - i)): #Bonferonni Holm
                print(f"corrected p = {p_val * (ntests - i)}")
                y_bar = y_max + spacing + (i * spacing * 2)
                x1, x2 = idx1, idx2
                
                plt.plot([x1, x2], [y_bar, y_bar], '-k', linewidth=2)
                plt.text((x1 + x2) / 2, y_bar + 0.005, '*', 
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize=20,
                        fontweight='bold')
            
                y_max = y_bar + spacing
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', format='svg')
        plt.close()
    
    print("\nStatistical Test Results:")
    if binary_comparison:
        print(f"Mann-Whitney U test: statistic={stat}, p-value={p_val:.4f}")
        print(f"Mean ± 95% CI:")
        for group in ['Engel I/RNS Responder', 'Other Outcomes']:
            data = df[df['Outcome'] == group]['Value']
            ci_lower, ci_upper = stats.bootstrap((data,), np.mean, 
                                               n_resamples=10000, 
                                               confidence_level=0.95).confidence_interval
            print(f"{group}: {data.mean():.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    return df
        
if __name__ == "__main__":

    results_file = "/home/sameer/all_location_restingSEEG_SOZ/all_soz_results/results_json/complete_results_1218_perm4_alllocs_v2.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    with open("outcomes_dict_v2_1yr_022625.json", 'r') as f:
        dict_outcomes = json.load(f)
    metrics_list = extract_patient_metrics(results)
    df = plot_patient_outcome_metrics(
            metrics_list,
            dict_outcomes,
            metric="Accuracy", #Youden's Index or Accuracy
            binary_comparison=False, 
            save_path='/home/sameer/all_location_restingSEEG_SOZ/figures/patient_outcomes_accuracy_1yr_v7.svg'
        )