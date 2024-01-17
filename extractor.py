# Q&A Chatbot
# from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image

import google.generativeai as genai

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load OpenAI model and get responses
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Chatbot", page_icon=":robot_face:")

# Main title and subtitle with instructions
st.title("Invoice Expert Chatbot")
st.subheader("Get answers about invoices from the Gemini Image Chatbot")
st.write("Upload a clear and visible invoice image for accurate results.")

# Image upload and user input section
st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

# User input for the chatbot
st.sidebar.header("User Input")
input = st.sidebar.text_input("Enter your prompt:")
submit = st.sidebar.button("Ask the Chatbot")

# Instructions for the user
st.sidebar.write("Instructions:")
st.sidebar.write("- Upload a clear and visible invoice image.")
st.sidebar.write("- Enter your prompt in the text box.")
st.sidebar.write("- Click 'Ask the Chatbot' to get a response.")

# If the Ask button is clicked
if submit:
    input_prompt = """
                   You are an expert in understanding invoices.
                   You will receive input images as invoices &
                   you will have to answer questions based on the input image
                   """
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is:")
    st.write(response)
