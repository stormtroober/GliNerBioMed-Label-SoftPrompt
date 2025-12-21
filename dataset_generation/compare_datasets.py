# -*- coding: utf-8 -*-
"""
Script per confrontare le differenze nella tokenizzazione tra il dataset Bi-Encoder e Mono-Encoder.
Analizza la lunghezza delle sequenze, il numero di token e mostra esempi visivi.
(Versione Dependency-Free)
"""
import json

PATH_BI = "dataset/dataset_tknlvl_bi.json"
PATH_MONO = "dataset/dataset_tknlvl_mono.json"

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean(data):
    return sum(data) / len(data) if data else 0

print(f"📥 Caricamento dataset...")
try:
    data_bi = load_data(PATH_BI)
    data_mono = load_data(PATH_MONO)
except FileNotFoundError as e:
    print(f"❌ Errore: {e}")
    exit(1)

print(f"📊 Dataset Bi-Encoder: {len(data_bi)} campioni")
print(f"📊 Dataset Mono-Encoder: {len(data_mono)} campioni")

if len(data_bi) != len(data_mono):
    print("⚠️ I dataset hanno un numero diverso di campioni!")

# Analisi statistiche
bi_lens = [len(x['tokens']) for x in data_bi]
mono_lens = [len(x['tokens']) for x in data_mono]

print(f"\n📏 Statistiche Lunghezza Token:")
print(f"   • Bi-Encoder  (Avg): {mean(bi_lens):.1f} | Max: {max(bi_lens)} | Min: {min(bi_lens)}")
print(f"   • Mono-Encoder (Avg): {mean(mono_lens):.1f} | Max: {max(mono_lens)} | Min: {min(mono_lens)}")

diffs = [m - b for m, b in zip(mono_lens, bi_lens)]
avg_diff = mean(diffs)
print(f"\n📉 Differenza media (Mono - Bi): {avg_diff:+.1f} token per frase")
print(f"   (Se positivo, il tokenizer del Mono spezza di più le parole)")

# Esempi visivi
print(f"\n🔍 Confronto Esempi (Primi 3):")
print("="*80)

for i in range(min(3, len(data_bi))):
    toks_bi = data_bi[i]['tokens']
    toks_mono = data_mono[i]['tokens']
    
    print(f"📌 Campione {i+1}:")
    print(f"   [Bi-Enc]  ({len(toks_bi)} tok): {toks_bi[:15]} ...")
    print(f"   [Mono-Enc]({len(toks_mono)} tok): {toks_mono[:15]} ...")
    print("-" * 80)

# Check label integrity
labels_bi = [l for x in data_bi for l in x['labels'] if l != -100]
labels_mono = [l for x in data_mono for l in x['labels'] if l != -100]

print(f"\n🏷️  Controllo Integrità Label (Escluso padding/special):")
print(f"   • Entità totali Bi-Encoder:   {len(labels_bi)}")
print(f"   • Entità totali Mono-Encoder: {len(labels_mono)}")

if len(labels_bi) != len(labels_mono):
    print("⚠️  ATTENZIONE: Il numero di label valide differisce! Verifica l'allineamento.")
    diff_labels = len(labels_bi) - len(labels_mono)
    print(f"   Differenza: {diff_labels} label")
else:
    print("✅ Il numero di token etichettati corrisponde esattamente.")
