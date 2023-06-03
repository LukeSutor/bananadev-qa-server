# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    ckpt = "mrm8488/longformer-base-4096-finetuned-squadv2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt)

    pipeline("question-answering", model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    download_model()