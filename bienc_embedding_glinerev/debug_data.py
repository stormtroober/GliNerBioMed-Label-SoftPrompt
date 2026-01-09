import json
import torch

DATASET_PATH = "../dataset/dataset_span_bi.json"
LABEL2ID_PATH = "../dataset/label2id.json"
LABEL2DESC_PATH = "../dataset/label2desc.json"

def load_and_map_labels(dataset, label2id_path, label2desc_path):
    print(f"Loading label mappings...")
    with open(label2id_path) as f: label2id = json.load(f)
    with open(label2desc_path) as f: label2desc = json.load(f)
    
    id2desc = {}
    for label_name, idx in label2id.items():
        id2desc[str(idx)] = label2desc[label_name]
        
    count = 0
    for item in dataset:
        for ner in item['ner']:
            original_label = ner[2] # "3"
            # N.B: original_label in JSON is typically string "3" or int 3?
            # Let's handle both strings and try to match
            if str(original_label) in id2desc:
                ner[2] = id2desc[str(original_label)]
                count += 1
            else:
                print(f"Warning: label {original_label} not found in id2desc keys: {list(id2desc.keys())[:3]}...")
                
    print(f"Replaced {count} labels.")
    return dataset

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

print("Mapping labels...")
data = load_and_map_labels(data, LABEL2ID_PATH, LABEL2DESC_PATH)

print("\nSAMPLE ITEM 0:")
print(json.dumps(data[0]['ner'], indent=2))

print("\nUnique labels found in first 100 items:")
labels = set()
for item in data[:100]:
    for ner in item['ner']:
        labels.add(ner[2])
for l in list(labels)[:5]:
    print(f" - {l[:50]}...")
