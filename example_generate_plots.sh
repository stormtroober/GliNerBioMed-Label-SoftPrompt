#!/bin/bash

# Script di esempio per generare grafici Optuna (PDF ottimizzati per Tesi/Pubblicazioni)
# Assicurati di aver attivato il venv prima di eseguire questo script

# Attiva il virtual environment
source ./venvgliner/bin/activate

# 1. Genera grafici PDF con titolo personalizzato (STANDARD USE CASE)
echo "=== Esempio 1: Genera grafici PDF per Tesi (Soft Prompting -> F1 Score) ==="
python generate_optuna_plots.py softprompting/optunas/jnlpa/results_20260202_194425.json \
  --metric "Validation Macro F1"

# 2. Genera solo grafici summary
echo -e "\n=== Esempio 2: Genera solo grafici Summary (Hard Prompting -> Loss) ==="
python generate_optuna_plots.py hardprompting/optunas/jnlpa/optuna_study_20260130_185904.json \
  --plot-type summary \
  --metric "Validation Loss"

# 3. Genera grafici in una directory specifica
echo -e "\n=== Esempio 3: Output in directory specifica ==="
python generate_optuna_plots.py softprompting/optunas/jnlpa/results_20260202_194425.json \
  --output-dir ./thesis_figures

# 4. Esempio con prefisso personalizzato
echo -e "\n=== Esempio 4: Prefisso personalizzato ==="
python generate_optuna_plots.py softprompting/optunas/jnlpa/results_20260202_194425.json \
  --prefix fig_3_results

# 5. Esempio con parametri booleani
echo -e "\n=== Esempio 5: Parametri Booleani ==="
if [ -f "example_boolean_params.json" ]; then
    python generate_optuna_plots.py example_boolean_params.json
else
    echo "⚠️ File example_boolean_params.json non trovato."
fi

echo -e "\n✅ Tutti gli esempi completati! I file sono pronti per essere inseriti nella tesi."
