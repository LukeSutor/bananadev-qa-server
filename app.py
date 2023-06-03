from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    ckpt = "mrm8488/longformer-base-4096-finetuned-squadv2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    mod = AutoModelForQuestionAnswering.from_pretrained(ckpt)

    model = pipeline("question-answering", model=mod, tokenizer=tokenizer, device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    question = model_inputs.get('question', None)
    text = model_inputs.get('text', None)
    if question is None:
        return {'message': "No question provided"}
    if text is None:
        return {'message': "No text provided"}
    
    # Run the model
    result = pipeline({"question": question, "context": text})['answer']

    # Return the results as a dictionary
    return result
