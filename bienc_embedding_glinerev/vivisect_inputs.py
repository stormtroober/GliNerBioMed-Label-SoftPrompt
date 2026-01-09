import torch
from gliner import GLiNER

def vivisect_label_processing():
    model_name = "Ihor/gliner-biomed-bi-small-v1.0"
    print(f"Loading model: {model_name}...")
    
    try:
        model = GLiNER.from_pretrained(model_name)
    except:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = GLiNER(config)

    print("\n=== LABEL TOKENIZATION VIVISECTION ===")
    
    # Define some sample labels
    labels = ["Person", "Organization", "DNA Structure"]
    print(f"Testing Labels: {labels}")
    
    # Access the internal label encoder model (BERT-based usually)
    # In Bi-GLiNER, labels_encoder usually wraps a Transformer
    lbl_encoder_model = model.model.token_rep_layer.labels_encoder.model
    # This model usually has a tokenizer if it's based on sentence-transformers, 
    # BUT GLiNER class handles tokenization centrally usually.
    # Let's see how GLiNER prepares labels.
    
    # We can inspect the 'predict_entities' method flow conceptually, but simpler:
    # GLiNER has a method to get label embeddings. Let's trace it.
    
    # Normally, GLiNER tokenizes labels using the same tokenizer as inputs? 
    # Or does it have a separate one? For 'all-MiniLM-L6-v2', it's likely BERT tokenizer.
    
    # Let's inspect the tokenizer directly attached to the model (if any)
    # The GLiNER class usually doesn't expose 'tokenizer' as a public attribute easily 
    # if it's hidden inside the label encoder.
    
    # However, we can look at the label encoder's 'tokenize' method if strictly used.
    # But let's act like we are providing inputs to the 'labels_encoder'.
    
    # Assuming standard GLiNER usage:
    # It converts labels to tokens.
    
    print("\nAttempting to tokenize labels using model's tokenizer...")
    
    # In GLiNER source, caching uses something like:
    # label_embeddings = model.compute_label_embeddings(labels) -- Does this exist? No?
    # model.set_sampling_params(...) 
    
    # Let's manually access the tokenizer found in the config or model
    # Usually model.data_processor or similar holds it?
    # Actually, GLiNER Bi-Encoder usually relies on the tokenizer being passed or loaded.
    
    # Let's try to find the tokenizer
    from transformers import AutoTokenizer
    # all-MiniLM-L6-v2 uses BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    for label in labels:
        print(f"\nLabel: '{label}'")
        # Tokenize (adding special tokens)
        encoded = tokenizer(label, add_special_tokens=True)
        ids = encoded['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        print(f"  IDs: {ids}")
        print(f"  Tokens: {tokens}")
        
        if tokens[0] == '[CLS]' and tokens[-1] == '[SEP]':
            print("  ✅ Format Confirmed: [CLS] ... [SEP]")
        else:
            print("  ❌ Format Unexpected!")

if __name__ == "__main__":
    vivisect_label_processing()
