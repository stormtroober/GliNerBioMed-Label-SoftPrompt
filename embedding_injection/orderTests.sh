#!/bin/bash

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Directory contenente i risultati dei test.
# Presuppone che lo script venga lanciato dalla root del progetto.
TARGET_DIR="./test_results"

# File temporaneo per memorizzare i punteggi
TEMP_FILE="scores_temp.txt"

# Controlla se la directory esiste
if [ ! -d "$TARGET_DIR" ]; then
    echo "Errore: La directory $TARGET_DIR non esiste."
    echo "Assicurati di lanciare lo script dalla root del progetto o modifica TARGET_DIR."
    exit 1
fi

echo "=== Inizio elaborazione in: $TARGET_DIR ==="
echo "=== Criterio: Media Armonica (Harmonic Mean) tra Macro e Micro F1 ==="

# ==========================================
# 1. PULIZIA NOMI PRECEDENTI
# ==========================================
echo "1. Ripristino nomi originali (rimozione prefissi vecchi)..."
cd "$TARGET_DIR" || exit

# Rimuove prefissi numerici esistenti (es. "1-eval..." torna "eval...")
for file in [0-9]*-*.md; do
    [ -e "$file" ] || continue
    new_name="${file#*-}" # Rimuove tutto fino al primo trattino
    mv "$file" "$new_name"
done

# ==========================================
# 2. ESTRAZIONE E CALCOLO PUNTEGGIO
# ==========================================
echo "2. Calcolo punteggi combinati..."

# Pulisce il file temporaneo (che si trova nella cartella superiore rispetto a test_results)
> "../$TEMP_FILE"

for file in *.md; do
    [ -e "$file" ] || continue

    # Estrae Macro F1
    macro_f1=$(grep "**Macro F1**" "$file" | awk -F'|' '{print $3}' | tr -d ' ')
    
    # Estrae Micro F1
    micro_f1=$(grep "**Micro F1**" "$file" | awk -F'|' '{print $3}' | tr -d ' ')

    # Gestione errori se i valori mancano
    if [ -z "$macro_f1" ]; then macro_f1="0"; fi
    if [ -z "$micro_f1" ]; then micro_f1="0"; fi

    # --- CALCOLO CRITERIO AVANZATO (MEDIA ARMONICA) ---
    # Formula: 2 * (Macro * Micro) / (Macro + Micro)
    # Premia i modelli bilanciati, penalizza se uno dei due valori è basso.
    score=$(awk -v ma="$macro_f1" -v mi="$micro_f1" 'BEGIN { 
        sum = ma + mi;
        if (sum > 0) printf "%.5f", (2 * ma * mi) / sum; 
        else print "0" 
    }')

    # Salva nel formato: SCORE MACRO MICRO NOMEFILE
    echo "$score $macro_f1 $micro_f1 $file" >> "../$TEMP_FILE"
done

# ==========================================
# 3. ORDINAMENTO E RINOMINA
# ==========================================
echo "3. Classifica e rinomina file..."

# Ordina in base allo SCORE (colonna 1) in ordine decrescente numerico (-nr)
rank=1
sort -k1,1nr "../$TEMP_FILE" | while read -r score macro micro filename; do
    
    # Se lo score è 0, probabilmente il file era vuoto o corrotto
    if [ "$score" == "0" ] || [ "$score" == "0.00000" ]; then
        echo "Skip (Score 0): $filename"
        continue
    fi

    # Costruisce il nuovo nome: CLASSIFICA-NomeOriginale
    new_filename="${rank}-${filename}"
    
    # Rinomina il file
    mv "$filename" "$new_filename"
    
    echo "[$rank] H-Mean: $score (Mac: $macro | Mic: $micro) -> $new_filename"
    
    ((rank++))
done

# ==========================================
# PULIZIA FINALE
# ==========================================
rm "../$TEMP_FILE"
echo "=== Completato! File ordinati per Media Armonica decrescente. ==="