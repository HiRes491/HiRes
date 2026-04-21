import tensorflow as tf
import os

def inspect_model(model_path):
    """Inspect a Keras model's architecture"""

    print(f"Loading model: {model_path}\n")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    model.summary()

    print("\n" + "=" * 60)
    print("MODEL DETAILS")
    print("=" * 60)

    print(f"\nInput shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")

    # Count trainable vs non-trainable
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")

    print("\n" + "=" * 60)
    print("LAYER BREAKDOWN")
    print("=" * 60)

    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

    print("\nLayer type counts:")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1]):
        print(f"  {layer_type}: {count}")

    print("\n" + "=" * 60)
    print("INPUT/OUTPUT INFO")
    print("=" * 60)
    print(f"\nThis model expects:")
    print(f"  - Input: RGB image of shape {model.input_shape[1:3]} (height x width)")
    print(f"  - Output: Segmentation mask with {model.output_shape[-1]} classes")

    # Visualize model to file (optional)
    try:
        plot_path = "model_architecture.png"
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # Top to bottom
            expand_nested=True,
            dpi=100
        )
        print(f"\nModel architecture diagram saved to: {plot_path}")
    except Exception as e:
        print(f"\nCould not save architecture diagram: {e}")
        print("Install graphviz and pydot for visualization: pip install pydot graphviz")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path = os.path.join(project_root, "models", "resistor_unet.keras")

    if os.path.exists(model_path):
        inspect_model(model_path)
    else:
        print(f"Model not found: {model_path}")

if __name__ == "__main__":
    main()
