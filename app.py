#This script contains the server side code of the chatbot, which is hosted on Flask

from flask_cors import CORS
from flask import Flask, request, jsonify
from agent import agent_executor
app = Flask(__name__)
CORS(app, resources={r"/ask_question": {"origins": "*"}})


@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    response = agent_executor.invoke({"input": question})
    # Extract the output field from the response
    output = response.get('output', 'No output available')
    return jsonify({"response": output})

if __name__ == '__main__':
    app.run(debug=True)
