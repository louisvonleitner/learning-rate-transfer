from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.save_pretrained(
    "/mnt/vast-nhr/projects/bthesis_louis_vonleitner/models/t5-base-local"
)
