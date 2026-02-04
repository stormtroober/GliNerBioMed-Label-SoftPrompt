
import torch
from gliner import GLiNER

def print_layer_summary(model):
    """
    Stampa una versione semplificata dei layer del modello GLiNER.
    """
    print("\n" + "="*60)
    print(f"🔍 INSPECTING MODEL: {model.__class__.__name__}")
    print("="*60)

    # Access the inner model structure
    # Based on the user's training script, GLiNER wraps a core model
    if hasattr(model, "model"):
        core = model.model
    else:
        core = model
    
    print(f"\n🔹 Core Model: {core.__class__.__name__}")
    
    # 1. Token Representation Layer
    if hasattr(core, 'token_rep_layer'):
        print("\n   📌 [1] Token Representation Layer")
        token_rep = core.token_rep_layer
        print(f"      Type: {token_rep.__class__.__name__}")
        
        # Text Encoder
        if hasattr(token_rep, 'bert_layer'):
            bert_layer = token_rep.bert_layer
            print(f"      Text Encoder Wrapper: {bert_layer.__class__.__name__}")
            
            # Try to resolve the underlying model
            if hasattr(bert_layer, 'model'):
                bert_model = bert_layer.model
                print(f"      Underlying Model: {bert_model.__class__.__name__}")
                
                # Try getting config
                if hasattr(bert_model, 'config'):
                    print(f"\n      Output Dimension (Hidden Size): {getattr(bert_model.config, 'hidden_size', 'N/A')}")
                
                print(f"\n      🧩 Text Encoder Architecture:")
                print(f"         {bert_model}")
            else:
                print(f"      Layer details: {bert_layer}")


        # Label Encoder (Bi-Encoder specific)
        if hasattr(token_rep, 'labels_encoder') and token_rep.labels_encoder is not None:
             print("\n      🧩 Label Encoder (for Bi-Encoder):")
             lbl_model = token_rep.labels_encoder.model
             if hasattr(lbl_model, 'encoder') and hasattr(lbl_model.encoder, 'layer'):
                 num_layers = len(lbl_model.encoder.layer)
                 print(f"         - Encoder: {num_layers} x Transformer Layers")
             else:
                 print(f"         - Model: {lbl_model}")
        
        # Projection (if present)
        if hasattr(token_rep, 'labels_projection'):
            print("\n      🔄 Projection Layer (Text -> Label Space):")
            print(f"         {token_rep.labels_projection}")

    # 2. Span Representation / Scoring
    # GLiNER uses span representations. Let's look for known components.
    print("\n   📌 [2] Other Components")
    for name, module in core.named_children():
        if name != 'token_rep_layer':
             print(f"      - {name}: {module}")

    print("="*60 + "\n")

def main():
    MODEL_NAME = "Ihor/gliner-biomed-bi-small-v1.0"
    
    print(f"📥 Loading {MODEL_NAME}...")
    try:
        model = GLiNER.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Inspect Layers
    print_layer_summary(model)

    # Example Usage
    text_example = "The patient was diagnosed with severe pneumonia and prescribed antibiotics."
    labels = ["Disease", "Treatment", "Symptom"]
    
    print(f"🧪 FUNCTIONAL TEST")
    print(f"   Input: '{text_example}'")
    print(f"   Labels: {labels}")
    
    entities = model.predict_entities(text_example, labels)
    
    print("\n   ✅ Output Entities:")
    for e in entities:
        print(f"      - {e['text']} ({e['label']}) | Score: {e['score']:.4f}")

if __name__ == "__main__":
    main()
