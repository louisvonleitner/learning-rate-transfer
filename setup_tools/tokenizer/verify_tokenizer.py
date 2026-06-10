import transformers as hftr

# Force offline mode to test isolation
import os

os.environ["HF_HUB_OFFLINE"] = "1"

try:
    tokenizer = hftr.T5TokenizerFast.from_pretrained(
        "/mnt/vast-nhr/projects/bthesis_louis_vonleitner/models/t5-base-local",
        model_max_length=1024,
    )
    print(
        "SUCCESS: Tokenizer loaded locally with max length:", tokenizer.model_max_length
    )
except Exception as e:
    print("FAILED:", e)
