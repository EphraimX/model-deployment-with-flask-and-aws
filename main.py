from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

def question_answer(context, question):

  result = question_answerer(question="What is a good example of a question answering dataset?", context=context)
  score = result['score']
  answer = result['answer']

  return score, answer

@app.route('/', methods=['POST', 'GET'])
def home():
  if request.method == 'POST':

    data = request.json
    context = data['context']
    question = data['question']

    score, answer = question_answer(context, question)
    score = round(score, 2)

    message = f"{question} Answer:  {answer.capitalize()} \n I am {score*100} percent sure of this"

    return message
  else:
    return "Hi, my name is MIA, I answer questions based on contexts."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
