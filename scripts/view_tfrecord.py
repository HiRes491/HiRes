import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_tfrecord(tfrecord_path, num_samples=5):
    """Parse and visualize samples from a TFRecord file"""

    print(f"Reading: {tfrecord_path}\n")

    # Create dataset from TFRecord
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # First, let's inspect the structure of one record
    print("=" * 50)
    print("TFRECORD STRUCTURE")
    print("=" * 50)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        print("\nFeatures found in TFRecord:")
        for key, feature in example.features.feature.items():
            # Determine feature type
            if feature.bytes_list.value:
                print(f"  - {key}: bytes (likely image data)")
            elif feature.float_list.value:
                print(f"  - {key}: float list (length: {len(feature.float_list.value)})")
            elif feature.int64_list.value:
                print(f"  - {key}: int64 list (values: {list(feature.int64_list.value)[:5]}...)")

    # Try to parse as image segmentation dataset
    # Common feature descriptions for segmentation datasets
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    print("\n" + "=" * 50)
    print(f"VISUALIZING {num_samples} SAMPLES")
    print("=" * 50)

    # Try different parsing strategies
    samples_shown = 0

    for i, raw_record in enumerate(raw_dataset.take(num_samples * 2)):  # Take extra in case some fail
        if samples_shown >= num_samples:
            break

        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            # Try to extract image and mask
            image = None
            mask = None

            for key, feature in example.features.feature.items():
                if feature.bytes_list.value:
                    data = feature.bytes_list.value[0]
                    try:
                        decoded = tf.io.decode_image(data).numpy()
                        if 'mask' in key.lower() or 'segmentation' in key.lower() or 'label' in key.lower():
                            mask = decoded
                        else:
                            image = decoded
                    except:
                        pass

            if image is not None:
                samples_shown += 1
                print(f"\nSample {samples_shown}:")
                print(f"  Image shape: {image.shape}")
                if mask is not None:
                    print(f"  Mask shape: {mask.shape}")
                    print(f"  Unique mask values: {np.unique(mask)[:15]}...")

                # Plot
                fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(10, 5))

                if mask is not None:
                    axes[0].imshow(image)
                    axes[0].set_title(f"Image {samples_shown}")
                    axes[0].axis('off')

                    axes[1].imshow(mask if len(mask.shape) == 3 else mask, cmap='tab20' if len(mask.shape) == 2 else None)
                    axes[1].set_title(f"Segmentation Mask {samples_shown}")
                    axes[1].axis('off')
                else:
                    if isinstance(axes, np.ndarray):
                        axes[0].imshow(image)
                        axes[0].set_title(f"Image {samples_shown}")
                        axes[0].axis('off')
                    else:
                        axes.imshow(image)
                        axes.set_title(f"Image {samples_shown}")
                        axes.axis('off')

                plt.tight_layout()
                plt.savefig(f"tfrecord_sample_{samples_shown}.png", dpi=100)
                plt.show()

        except Exception as e:
            print(f"  Could not parse record {i}: {e}")

    # Count total records
    total = sum(1 for _ in raw_dataset)
    print(f"\n" + "=" * 50)
    print(f"SUMMARY")
    print(f"=" * 50)
    print(f"Total records in file: {total}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    tfrecord_dir = os.path.join(project_root, "data", "tfrecords")

    # List TFRecord files
    tfrecord_files = [f for f in os.listdir(tfrecord_dir) if 'tfrecord' in f.lower()]

    print("Available TFRecord files:")
    for i, f in enumerate(tfrecord_files):
        print(f"  {i+1}. {f}")

    # Parse training file
    training_file = os.path.join(tfrecord_dir, "training.tfrecord-0-1")
    if os.path.exists(training_file):
        parse_tfrecord(training_file, num_samples=3)

if __name__ == "__main__":
    main()
