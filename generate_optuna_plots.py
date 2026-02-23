#!/usr/bin/env python3
"""
Script per generare automaticamente grafici di visualizzazione Optuna da file JSON.

Genera due tipi di grafici:
1. Standard plots: Optimization History + Hyperparameter Importance
2. Summary plots: Optimization History + scatter plots per i parametri principali

Usage:
    python generate_optuna_plots.py <json_file>
    python generate_optuna_plots.py <json_file> --output-dir <dir>
    python generate_optuna_plots.py <json_file> --plot-type summary
    python generate_optuna_plots.py <json_file> --plot-type standard
    python generate_optuna_plots.py <json_file> --plot-type both
"""

import json
import argparse
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.ticker import MaxNLocator


def load_optuna_json(json_path: str) -> Dict[str, Any]:
    """Carica il file JSON dello studio Optuna."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_trials(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Restituisce la lista di trial dal JSON, supportando sia 'all_trials' che 'trials' come chiave."""
    if 'all_trials' in data:
        return data['all_trials']
    elif 'trials' in data:
        return data['trials']
    return []


def extract_completed_trials(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Estrae solo i trial completati (non pruned)."""
    return [
        trial for trial in get_trials(data)
        if trial['state'] == 'TrialState.COMPLETE' and trial['value'] is not None
    ]


def get_param_names(trials: List[Dict[str, Any]]) -> List[str]:
    """Ottiene i nomi di tutti i parametri dai trial."""
    if not trials:
        return []
    return list(trials[0]['params'].keys())


def calculate_param_importance(trials: List[Dict[str, Any]], param_names: List[str]) -> Dict[str, float]:
    """
    Calcola l'importanza dei parametri usando la correlazione con i valori obiettivo.
    Questo è un metodo semplificato; Optuna usa metodi più sofisticati.
    """
    if len(trials) < 2:
        return {param: 0.0 for param in param_names}
    
    values = np.array([trial['value'] for trial in trials])
    importance = {}
    
    for param in param_names:
        param_values = []
        for trial in trials:
            val = trial['params'][param]
            # Converti batch_size categorico in numerico se necessario
            if isinstance(val, str):
                try:
                    val = float(val)
                except:
                    # Per parametri categorici, usa l'indice
                    unique_vals = sorted(set(t['params'][param] for t in trials))
                    val = unique_vals.index(val)
            param_values.append(val)
        
        param_values = np.array(param_values)
        
        # Calcola correlazione assoluta
        if len(set(param_values)) > 1:  # Solo se il parametro varia
            correlation = abs(np.corrcoef(param_values, values)[0, 1])
            importance[param] = correlation if not np.isnan(correlation) else 0.0
        else:
            importance[param] = 0.0
    
    # Normalizza le importanze
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}
    
    return importance


def plot_optimization_history(ax, trials: List[Dict[str, Any]], best_value: float, metric_name: str = "Objective Value"):
    """Plotta la storia dell'ottimizzazione."""
    trial_numbers = [trial['number'] for trial in trials]
    values = [trial['value'] for trial in trials]
    
    ax.plot(trial_numbers, values, 'o-', linewidth=3, markersize=10, alpha=0.7)
    ax.axhline(y=best_value, color='r', linestyle='--', linewidth=3, 
               label=f'Best: {best_value:.4f}', alpha=0.7)
    ax.set_xlabel('Trial Number', fontsize=18, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')
    ax.set_title('Optimization History', fontsize=20, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=16)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_hyperparameter_importance(ax, importance: Dict[str, float]):
    """Plotta l'importanza degli iperparametri."""
    # Ordina per importanza
    sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    params = [p[0] for p in sorted_params]
    values = [p[1] for p in sorted_params]
    
    # Determina dimensione font in base alla lunghezza massima dei nomi
    max_param_length = max(len(p) for p in params) if params else 0
    if max_param_length > 25:
        param_fontsize = 10
    elif max_param_length > 20:
        param_fontsize = 11
    elif max_param_length > 15:
        param_fontsize = 13
    else:
        param_fontsize = 16
    
    y_pos = np.arange(len(params))
    bars = ax.barh(y_pos, values, alpha=0.8, color='steelblue', edgecolor='navy', linewidth=1.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=param_fontsize)
    ax.set_xlabel('Importance', fontsize=18, fontweight='bold')
    ax.set_title('Hyperparameter Importance', fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.tick_params(axis='x', labelsize=16)
    
    # Aggiungi valori sulle barre
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=14, fontweight='bold')
        
    ax.set_xlim(0, max(values) * 1.25)  # Più spazio per le etichette
    ax.invert_yaxis()


def plot_param_vs_objective(ax, trials: List[Dict[str, Any]], param_name: str, 
                           log_scale: bool = False, metric_name: str = "Objective Value"):
    """Plotta un parametro vs il valore obiettivo."""
    param_values = []
    objective_values = []
    is_boolean = False
    is_categorical = False
    categorical_mapping = {}
    
    # Prima passata: determina il tipo di parametro
    sample_val = trials[0]['params'][param_name]
    if isinstance(sample_val, str):
        # Controlla se è booleano o categorico
        unique_vals = set(trial['params'][param_name] for trial in trials)
        if all(v.lower() in ['true', 'false'] for v in unique_vals):
            is_boolean = True
        else:
            is_categorical = True
            # Crea mapping per parametri categorici
            categorical_mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
    
    for trial in trials:
        val = trial['params'][param_name]
        
        # Gestisci valori booleani
        if isinstance(val, bool):
            is_boolean = True
            param_values.append(1.0 if val else 0.0)
            objective_values.append(trial['value'])
        elif isinstance(val, str):
            # Prova a convertire stringhe booleane
            if val.lower() in ['true', 'false']:
                is_boolean = True
                param_values.append(1.0 if val.lower() == 'true' else 0.0)
                objective_values.append(trial['value'])
            elif is_categorical:
                # Usa il mapping categorico
                param_values.append(categorical_mapping[val])
                objective_values.append(trial['value'])
            else:
                try:
                    val = float(val)
                    param_values.append(val)
                    objective_values.append(trial['value'])
                except:
                    continue
        else:
            param_values.append(val)
            objective_values.append(trial['value'])
    
    if not param_values:
        ax.text(0.5, 0.5, f'No data for {param_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        return
    
    # Gestisci parametri categorici con box plot
    if is_categorical:
        # Raggruppa i valori per categoria
        category_data = {cat: [] for cat in categorical_mapping.keys()}
        for trial in trials:
            cat = trial['params'][param_name]
            category_data[cat].append(trial['value'])
        
        # Crea box plot
        categories = sorted(categorical_mapping.keys())
        data_to_plot = [category_data[cat] for cat in categories]
        
        bp = ax.boxplot(data_to_plot, tick_labels=categories, patch_artist=True,
                       widths=0.6, showmeans=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                       medianprops=dict(color='red', linewidth=3),
                       meanprops=dict(marker='D', markerfacecolor='green', markersize=10),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        # Aggiungi scatter plot sovrapposto
        for i, cat in enumerate(categories):
            y_vals = category_data[cat]
            x_vals = [i + 1] * len(y_vals)
            # Aggiungi jitter per visualizzare meglio i punti
            x_jitter = np.random.normal(0, 0.04, len(x_vals))
            ax.scatter(np.array(x_vals) + x_jitter, y_vals, 
                      alpha=0.6, s=100, c='navy', edgecolors='black', linewidth=1, zorder=3)
        
        ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=18, fontweight='bold')
        ax.set_xticklabels(categories, fontsize=14, rotation=15, ha='right')
        ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')
        ax.set_title(f'{param_name.replace("_", " ").title()} vs {metric_name}', 
                    fontsize=20, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(axis='y', which='major', labelsize=16)
        return
    
    # Usa un colormap per rappresentare l'ordine dei trial
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    scatter = ax.scatter(param_values, objective_values, c=colors, 
                        s=200, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Gestisci parametri booleani
    if is_boolean:
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(['False', 'True'], fontsize=16)
        ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=18, fontweight='bold')
        ax.set_xlim(-0.2, 1.2)
    elif log_scale and min(param_values) > 0:
        ax.set_xscale('log')
        ax.set_xlabel(f'{param_name} (log scale)', fontsize=18, fontweight='bold')
    else:
        ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=18, fontweight='bold')
    
    ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')
    ax.set_title(f'{param_name.replace("_", " ").title()} vs {metric_name}', 
                fontsize=20, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=16)


def generate_standard_plots(data: Dict[str, Any], output_path: str, metric_name: str = "Objective Value"):
    """Genera i grafici standard: Optimization History + Hyperparameter Importance."""
    trials = extract_completed_trials(data)
    
    if not trials:
        print("⚠️  Nessun trial completato trovato!")
        return
    
    param_names = get_param_names(trials)
    importance = calculate_param_importance(trials, param_names)
    
    # Estrai best_value da entrambi i formati
    if 'best_trial' in data:
        best_value = data['best_trial']['value']
    else:
        best_value = data.get('best_value', min(t['value'] for t in trials if t['value'] is not None))
    
    # Crea figura con 2 subplot affiancati
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    plot_optimization_history(ax1, trials, best_value, metric_name)
    plot_hyperparameter_importance(ax2, importance)
    
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.12, right=0.95, hspace=0.3, wspace=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Grafici standard salvati in: {output_path}")


def generate_summary_plots(data: Dict[str, Any], output_path: str, metric_name: str = "Objective Value"):
    """Genera i grafici summary: Optimization History + scatter plots per parametri chiave."""
    trials = extract_completed_trials(data)
    
    if not trials:
        print("⚠️  Nessun trial completato trovato!")
        return
    
    param_names = get_param_names(trials)
    
    # Estrai best_value da entrambi i formati
    if 'best_trial' in data:
        best_value = data['best_trial']['value']
    else:
        best_value = data.get('best_value', min(t['value'] for t in trials if t['value'] is not None))
    
    # Identifica i parametri da plottare (max 3 scatter plots)
    # Priorità: learning_rate, lr_scheduler_type, temperature, gamma, alpha, weight_decay, batch_size
    priority_params = ['learning_rate', 'lr_embed', 'lr_scheduler_type', 'temperature', 
                      'gamma', 'alpha', 'weight_decay', 'focal_gamma', 'focal_alpha', 
                      'focal_loss_gamma']
    
    params_to_plot = []
    for param in priority_params:
        if param in param_names:
            params_to_plot.append(param)
        if len(params_to_plot) == 3:
            break
    
    # Se non abbiamo abbastanza parametri prioritari, aggiungi altri
    if len(params_to_plot) < 3:
        for param in param_names:
            if param not in params_to_plot:
                params_to_plot.append(param)
            if len(params_to_plot) == 3:
                break
    
    # Crea figura con 2x2 grid
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Plot 1: Optimization History (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_optimization_history(ax1, trials, best_value, metric_name)
    
    # Plot 2-4: Scatter plots per i parametri selezionati
    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]
    
    log_scale_params = ['learning_rate', 'lr_embed', 'weight_decay']
    
    for i, param in enumerate(params_to_plot[:3]):
        if i < len(axes):
            use_log = param in log_scale_params
            plot_param_vs_objective(axes[i], trials, param, log_scale=use_log, metric_name=metric_name)
    
    # Se abbiamo meno di 3 parametri, rimuovi gli assi vuoti
    for i in range(len(params_to_plot), 3):
        if i < len(axes):
            axes[i].axis('off')
    
    # Aggiusta layout per testi grandi
    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.10, right=0.95, hspace=0.4, wspace=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Grafici summary salvati in: {output_path}")


def print_study_info(data: Dict[str, Any]):
    """Stampa informazioni sullo studio."""
    print("\n" + "="*70)
    print(f"📊 OPTUNA STUDY: {data.get('study_name', 'Unknown')}")
    print("="*70)
    print(f"⏱️  Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"🎯 Total trials: {data.get('n_trials', 0)}")
    
    # Calcola completed e pruned trials se non presenti
    all_t = get_trials(data)
    if all_t:
        completed = sum(1 for t in all_t if t.get('state') == 'TrialState.COMPLETE')
        pruned = sum(1 for t in all_t if t.get('state') == 'TrialState.PRUNED')
        print(f"✅ Completed trials: {data.get('completed_trials', completed)}")
        print(f"✂️  Pruned trials: {data.get('pruned_trials', pruned)}")
    else:
        print(f"✅ Completed trials: {data.get('completed_trials', 0)}")
        print(f"✂️  Pruned trials: {data.get('pruned_trials', 0)}")
    
    print(f"⏳ Total time: {data.get('total_time_seconds', 0):.2f} seconds")
    print(f"\n🏆 BEST TRIAL:")
    
    # Gestisci entrambi i formati: best_trial (completo) o best_params/best_value (semplificato)
    if 'best_trial' in data:
        print(f"   Trial number: {data['best_trial']['number']}")
        print(f"   Best value: {data['best_trial']['value']:.6f}")
        print(f"   Parameters:")
        for param, value in data['best_trial']['params'].items():
            print(f"      - {param}: {value}")
    elif 'best_params' in data and 'best_value' in data:
        trial_num = data.get('best_trial_number', 'N/A')
        print(f"   Trial number: {trial_num}")
        print(f"   Best value: {data['best_value']:.6f}")
        print(f"   Parameters:")
        for param, value in data['best_params'].items():
            print(f"      - {param}: {value}")
    else:
        print("   ⚠️  Best trial information not available")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Genera grafici di visualizzazione Optuna da file JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  %(prog)s study.json
  %(prog)s study.json --output-dir ./plots
  %(prog)s study.json --plot-type summary
  %(prog)s study.json --plot-type both
        """
    )
    
    parser.add_argument('json_file', type=str, help='Path al file JSON dello studio Optuna')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Directory di output (default: stessa directory del JSON)')
    parser.add_argument('--plot-type', '-t', type=str, 
                       choices=['standard', 'summary', 'both'], default='both',
                       help='Tipo di grafici da generare (default: both)')
    parser.add_argument('--prefix', '-p', type=str, default=None,
                       help='Prefisso per i nomi dei file di output')
    parser.add_argument('--metric', '-m', type=str, default=None,
                       help='Nome della metrica da visualizzare sugli assi (es. "Loss", "F1 Score"). Default: auto-detect.')
    
    args = parser.parse_args()
    
    # Verifica che il file esista
    if not os.path.exists(args.json_file):
        print(f"❌ Errore: File non trovato: {args.json_file}")
        return 1
    
    # Carica i dati
    print(f"📂 Caricamento file: {args.json_file}")
    data = load_optuna_json(args.json_file)
    
    # Stampa informazioni sullo studio
    print_study_info(data)
    
    # Determina la directory di output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.json_file).parent
    
    # Determina il prefisso per i file di output
    if args.prefix:
        prefix = args.prefix
    else:
        timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        prefix = f"optuna_{timestamp}"
    
    # Determina il nome della metrica
    metric_name = "Objective Value"
    if args.metric:
        metric_name = args.metric
    else:
        # Euristica semplice per indovinare il nome
        if 'best_trial' in data and data['best_trial']['value'] > 1.0:
            metric_name = "Loss"
        elif 'best_value' in data and data['best_value'] > 1.0:
             metric_name = "Loss"
        elif 'f1' in str(data.get('study_name', '')).lower():
            metric_name = "F1 Score"
    
    # Genera i grafici richiesti
    if args.plot_type in ['standard', 'both']:
        standard_path = output_dir / f"{prefix}_plots.pdf"
        print(f"🎨 Generazione grafici standard (PDF)...")
        generate_standard_plots(data, str(standard_path), metric_name=metric_name)
    
    if args.plot_type in ['summary', 'both']:
        summary_path = output_dir / f"{prefix}_summary.pdf"
        print(f"🎨 Generazione grafici summary (PDF)...")
        generate_summary_plots(data, str(summary_path), metric_name=metric_name)
    
    print("\n✨ Completato!")
    return 0


if __name__ == '__main__':
    exit(main())
