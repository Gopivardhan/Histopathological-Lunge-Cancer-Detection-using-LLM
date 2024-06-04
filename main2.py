import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import openai

# Load your model
Test_model = load_model("20230918-17581695059896-98%-predict accuracy.h5")  # your model file
class_names = ["Adenocarcinoma", "Benign_Tissue", "Squamous cell"]  # actual class names

# Set your OpenAI API key
openai.api_key = "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"

def predict(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_tensor = np.expand_dims(img_array, axis=0)
    img_tensor /= 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = Test_model.predict(img_tensor)

    # Get the predicted class index and class label
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return predicted_class, predictions[0]
# Model to access the api key of openai gpt  
def generate_description(predicted_class):
    prompt = f"Generate a description for the image with predicted class of Lung Cancer: {predicted_class}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt, # to avoid the version difference error
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response['choices'][0]['text'].strip()

def main():
    st.title("Lung Cancer Prediction with Generative AI")

    uploaded_file = st.file_uploader("Upload the image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make predictions
        predicted_class, probabilities = predict(uploaded_file)

        # Print predictions
        st.subheader(f"Predicted Class: {predicted_class}")
        st.subheader("Predicted Probabilities:")
        for class_label, probability in zip(class_names, probabilities):
            st.write(f"  {class_label}: {probability:.4f}")

        # Horizontal bar graph for predicted probabilities
        st.bar_chart(probabilities)

        # Generate description using GPT-3
        description = generate_description(predicted_class)
        st.subheader("Generated Description:")
        st.write(description)

if __name__ == "__main__":
    main()
