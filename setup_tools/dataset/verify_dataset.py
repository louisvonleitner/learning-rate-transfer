import os
from datasets import load_dataset

# Optional: Force Hugging Face to use a specific cache directory if your home directory is small
# os.environ["HF_DATASETS_CACHE"] = "/path/to/large/cluster/storage"


def test_c4_loading():
    print("Initializing C4 stream test...")

    try:
        # We specify the 'en' split and use streaming=True to avoid massive downloads
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

        print("Successfully connected to the dataset stream.")
        print("Fetching the first 3 samples to verify format:\n" + "=" * 50)

        # Take the first 3 elements from the stream iterator
        for i, sample in enumerate(dataset.take(3)):
            print(f"\n--- Sample {i+1} ---")
            print(f"URL: {sample.get('url')}")
            print(f"Timestamp: {sample.get('timestamp')}")
            # Truncate text output just to keep the terminal clean
            text_preview = sample.get("text", "")[:200].replace("\n", " ")
            print(f"Text Preview: {text_preview}...")

        print("\n" + "=" * 50)
        print("Test passed! Dataset structure is intact.")

    except Exception as e:
        print(f"\nAn error occurred while loading the dataset: {e}")


if __name__ == "__main__":
    test_c4_loading()
