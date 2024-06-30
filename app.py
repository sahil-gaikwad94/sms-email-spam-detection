from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
nb_classifier = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    transformed_message = vectorizer.transform([message])
    prediction = nb_classifier.predict(transformed_message)
    result = 'spam' if prediction[0] == 1 else 'ham'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
