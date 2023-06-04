from potassium import Potassium, Request, Response

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    ckpt = "mrm8488/longformer-base-4096-finetuned-squadv2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt)

    pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
   
    context = {
        "model": pipe
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    question = request.json.get("question")
    text = request.json.get("text")
    if question is None or text is None:
        return Response(
            json={"text": "Question and text fields required"}, 
            status=200
        )
    model = context.get("model")
    output = model({"question": question, "context": text})

    return Response(
        json = {"output": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()