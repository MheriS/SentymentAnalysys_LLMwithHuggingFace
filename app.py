from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Membuat pipeline untuk analisis sentimen dengan from_pt saat memuat model
classifier = pipeline("text-classification", model="ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa", tokenizer="ayameRushia/roberta-base-indonesian-1.5G-sentiment-analysis-smsa")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        texts = request.form.getlist('texts')  # Ambil teks dari form
        results = classifier(texts)
        result = [(text, res['label'], res['score']) for text, res in zip(texts, results)]
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error: {e}")