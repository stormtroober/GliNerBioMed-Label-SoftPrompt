import json
import os
from tqdm import tqdm

def reconstruct_and_convert(data_path, label2id, output_name):
    print(f"Processing {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Invert label2id to get id2label
    id2label = {v: k for k, v in label2id.items()}

    converted_data = []

    for item in tqdm(data, desc=f"Converting {output_name}"):
        tokens = item['tokens']
        # Check which key holds the labels. dataset_tokenlevel_simple usually uses 'labels' or 'ner_tags'
        tag_ids = item.get('labels', item.get('ner_tags'))
        
        if tag_ids is None:
            # If no labels found, skip or handle appropriately
            continue

        ner_spans = [] # List of [start, end, label_id]
        
        words = []
        word_to_token_indices = [] # Map word index -> list of original token indices
        
        current_word_tokens = []
        current_word_original_indices = []
        
        for i, t in enumerate(tokens):
            if t == "[CLS]" or t == "[SEP]" or t == "<pad>":
                continue
                
            # Check for start of new word
            # SentencePiece usually uses "_" (U+2581) at current position 0
            is_start = t.startswith(" ") or t.startswith("▁") # Checking both just in case
            
            clean_token = t.replace(" ", "").replace("▁", "")
            
            if is_start:
                if current_word_tokens:
                    words.append("".join(current_word_tokens))
                    word_to_token_indices.append(current_word_original_indices)
                    
                current_word_tokens = [clean_token]
                current_word_original_indices = [i]
            else:
                if not current_word_tokens:
                    current_word_tokens = [clean_token]
                    current_word_original_indices = [i]
                else:
                    current_word_tokens.append(clean_token)
                    current_word_original_indices.append(i)
        
        # Flush last word
        if current_word_tokens:
            words.append("".join(current_word_tokens))
            word_to_token_indices.append(current_word_original_indices)
            
        
        # Now we need to map the ner tags (which are per original token) to the new words
        active_entity = None # {label_id, start_word_idx}
        
        # Iterate over constructed words and check their original tokens' labels
        for w_idx, original_indices in enumerate(word_to_token_indices):
            # Get labels for these tokens
            # We take the label of the FIRST token of the word as the ground truth for the word
            
            first_token_idx = original_indices[0]
            if first_token_idx >= len(tag_ids):
                continue
                
            tid = tag_ids[first_token_idx]
            
            # Check if valid ID
            if tid != -100 and tid in id2label:
                # Use the ID string as label
                # Note: We include 'O' (ID 5 usually) here as a span if it's in label2id.
                # Downstream scripts can filter 'O' if needed by checking the ID mapping.
                
                label_id_str = str(tid)
                
                if active_entity:
                    if active_entity['label'] == label_id_str:
                        # Continue entity
                        pass
                    else:
                        # Close previous, start new
                        ner_spans.append([active_entity['start'], w_idx - 1, active_entity['label']])
                        active_entity = {'label': label_id_str, 'start': w_idx}
                else:
                    # Start new
                    active_entity = {'label': label_id_str, 'start': w_idx}
            else:
                # It is ignored (-100) or invalid
                if active_entity:
                    # Close entity
                    ner_spans.append([active_entity['start'], w_idx - 1, active_entity['label']])
                    active_entity = None
        
        # Flush last
        if active_entity:
             ner_spans.append([active_entity['start'], len(words) - 1, active_entity['label']])
        
        converted_data.append({
            "tokenized_text": words,
            "ner": ner_spans
        })

    # Save
    with open(f"finetune/{output_name}", 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(converted_data)} items to finetune/{output_name}")


def main():
    base_path = "/home/aless/Desktop/GliNerBioMed-Label-SoftPrompt"
    
    # Load mappings
    with open(os.path.join(base_path, "label2id.json"), 'r') as f:
        label2id = json.load(f)
    print(f"Loaded {len(label2id)} labels.")
    
    # Convert Train
    reconstruct_and_convert(
        os.path.join(base_path, "dataset/dataset_tokenlevel_simple.json"),
        label2id,
        "jnlpa_train.json"
    )

    # Convert Test
    reconstruct_and_convert(
        os.path.join(base_path, "dataset/test_dataset_tokenlevel.json"),
        label2id,
        "jnlpa_test.json"
    )

if __name__ == "__main__":
    main()
