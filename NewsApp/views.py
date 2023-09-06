from django.shortcuts import render
from django.http import HttpResponse
from joblib import load
from django.contrib import messages
from keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load('./savedmodels/model.joblib')
tokenizer = load('./savedmodels/tokenizer.joblib')

# Define the preprocess_text function
def preprocess_text(text):
    # Your preprocessing code for the text here
    # Replace the following line with your actual preprocessing code
    preprocessed_text = text.lower()  # Example: convert to lowercase
    return preprocessed_text

# Define the preprocess_title function
def preprocess_title(title):
    # Your preprocessing code for the title here
    # Replace the following line with your actual preprocessing code
    preprocessed_title = title.lower()  # Example: convert to lowercase
    return preprocessed_title

# Define the max_sequence_length
max_sequence_length = 100

def predict_news(request):
    if request.method == 'POST':
        title = request.POST.get('title', '')
        text = request.POST.get('text', '')
        
        if text:
            # Preprocess the input text
            preprocessed_text = preprocess_text(text)
            
            # Tokenize and pad the input text
            text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
            text_padded = pad_sequences(text_sequence, maxlen=max_sequence_length)
            
            # Make the prediction
            predicted_label = model.predict(text_padded)[0][0]
            
            # Determine the output label
            if predicted_label >= 0.5:
                output_label = 'Fake'
            else:
                output_label = 'Real'
            
            return render(request, 'result.html', {'predicted_label': output_label})
        
        else:
            messages.error(request, "Please enter text.")
    
    return render(request, 'predict.html')
