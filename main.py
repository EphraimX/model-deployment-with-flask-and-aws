from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

def question_answer(context, question):

  result = question_answerer(question="What is a good example of a question answering dataset?", context=context)
  score = result['score']
  answer = result['answer']

  return score, answer


@app.route('/', methods=['GET'])
def home():
  return "Hi, I'm Mia, I answer questions based on a certain concept. \n \
        You can make a POST request to the 'ask-question' endpoint and see how I work. Thank you."


@app.route('/ask-question', methods=['POST', 'GET'])
def ask_questions():

  if request.method == 'POST':

    data = request.json
    context = data['context']
    question = data['question']

    score, answer = question_answer(context, question)
    score = round(score, 2)

    message = f"{question} Answer:  {answer.capitalize()} \n I am {score*100} percent sure of this"

    return message
  else:
    return "This Endpoint Only Accepts GET requests"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)