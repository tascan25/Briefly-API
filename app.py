# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize the summarization pipeline
summarizer = pipeline('summarization', model="facebook/bart-large-cnn")

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # The summarizer might have limitations on input length
    # So we might need to split the text if it's too long
    max_chunk = 500  # Adjust based on the model's max input length
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summaries = []
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = ' '.join(summaries)

    return jsonify({'summary': final_summary})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
# app.py

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import pipeline

# app = Flask(__name__)
# CORS(app)

# # Initialize the summarization pipeline
# summarizer = pipeline('summarization', model="facebook/bart-large-cnn")

# def summarize_text_chunk(text_chunk):
#     # Calculate input length in tokens
#     input_length = len(text_chunk.split())
    
#     # Set max_length to a percentage (e.g., 50%) of the input length
#     max_length = int(input_length * 0.5)
#     # Ensure max_length is at least 30 and doesn't exceed model limits
#     max_length = max(30, min(max_length, 130))
    
#     # Set min_length similarly
#     min_length = int(max_length * 0.5)
#     min_length = max(10, min(min_length, max_length - 1))
    
#     # Generate summary
#     summary = summarizer(
#         text_chunk,
#         max_length=max_length,
#         min_length=min_length,
#         do_sample=False
#     )
#     return summary[0]['summary_text']

# @app.route('/summarize', methods=['POST'])
# def summarize_text():
#     data = request.get_json()
#     text = data.get('text', '')

#     if not text:
#         return jsonify({'error': 'No text provided'}), 400

#     max_chunk_length = 500  # Adjust based on model's max input length
#     # Split the text into chunks without cutting words
#     words = text.split()
#     text_chunks = [' '.join(words[i:i+max_chunk_length]) for i in range(0, len(words), max_chunk_length)]

#     summaries = []
#     for chunk in text_chunks:
#         summary = summarize_text_chunk(chunk)
#         summaries.append(summary)

#     final_summary = ' '.join(summaries)

#     return jsonify({'summary': final_summary})

# if __name__ == '__main__':
#     app.run(debug=True)
