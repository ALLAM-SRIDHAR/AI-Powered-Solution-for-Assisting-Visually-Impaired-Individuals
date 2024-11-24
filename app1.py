import streamlit as st
from PIL import Image
import io
import base64
import google.generativeai as genai
import pytesseract
from gtts import gTTS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import shutil
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    st.error("Tesseract OCR is not installed or not found in PATH.")
else:
    st.write(f"Tesseract found at: {tesseract_path}")


# Configure Google API
#f = open('api key.txt')
#GOOGLE_API_KEY = f.read()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

# Configure Tesseract path - Replace this with your actual Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def init_gemini_vision():
    """Initialize the Gemini Vision model"""
    return genai.GenerativeModel("models/gemini-1.5-flash")

def init_gemini_text():
    """Initialize the text-based model"""
    return GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

def get_image_description(image_data, model):
    """Generate a detailed description of the image content"""
    prompt = """
    You are assisting a visually impaired person. Describe the image in detail, focusing on:
    1. The main elements and their spatial relationship
    2. Any potential hazards or obstacles
    3. Important text or signage
    4. Colors and lighting conditions
    Provide the description in a clear, organized manner.
    """
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    response = model.generate_content([prompt, image_parts[0]])
    return response.text

def extract_text_from_image(image):
    """Extract text from the image using OCR"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        cleaned_text = text.strip()
        if cleaned_text:
            return cleaned_text
        return "No text detected in the image."
    except Exception as e:
        return f"Error in text extraction: {str(e)}\nPlease make sure Tesseract is properly installed and the path is correct."

def detect_objects(image_data, model):
    """Detect objects and potential obstacles in the image"""
    prompt = """
    Analyze this image for a visually impaired person, focusing on:
    1. List all objects and their locations
    2. Identify potential obstacles or hazards
    3. Describe spatial relationships between objects
    4. Highlight any moving elements or changes in elevation
    Provide clear, safety-focused guidance.
    """
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    response = model.generate_content([prompt, image_parts[0]])
    return response.text

def get_task_specific_guidance(image_description, llm):
    """Generate task-specific guidance based on the image content"""
    prompt_template = PromptTemplate(
        input_variables=["description"],
        template="""
        Based on this image description: {description}
        
        Provide practical guidance for a visually impaired person, including:
        1. Safety considerations
        2. Navigation suggestions
        3. Relevant context about the environment
        4. Specific tips for interacting with identified objects
        
        Format the response in clear, concise paragraphs.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(description=image_description)

def text_to_speech(text, language='en'):
    """Convert text to speech and return the audio file as base64"""
    try:
        tts = gTTS(text=text, lang=language)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
        return audio_base64
    except Exception as e:
        return None

def create_accessible_button(label, key):
    """Create a large, accessible button with custom styling"""
    return st.button(
        label,
        key=key,
        help=f"Click to {label}",
        use_container_width=True
    )

def main():
    st.set_page_config(page_title="Vision Assistant", layout="wide")
    
    # Custom CSS for larger text and buttons
    st.markdown("""
        <style>
        .big-font {
            font-size:40px !important;
            margin-bottom: 20px;
        }
        .stButton>button {
            height: 320px; /* Increased height */
            width: 100%; /* Make button fill container */
            font-size: 28px; /* Increased font size */
            padding: 15px; /* Added padding */
            margin: 20px 0; /* Adjust spacing between buttons */
            background-color: #0066cc; /* Button background color */
            color: white; /* Button text color */
            border-radius: 12px; /* Rounded corners */
            border: none; /* Remove border */
        }
        .output-text {
            font-size: 20px;
            line-height: 1.5;
            padding: 15px;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">AI Vision Assistant</p>', unsafe_allow_html=True)
    
    # Initialize models
    vision_model = init_gemini_vision()
    text_model = init_gemini_text()
    
    st.markdown('<p class="big-font">Upload an Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        scene_button = create_accessible_button("📷 Analyze Scene", "scene")
        text_button = create_accessible_button("📝 Read Text", "text")
        objects_button = create_accessible_button("🔍 Detect Objects", "objects")
        guidance_button = create_accessible_button("💡 Get Assistance", "guidance")
        
        if scene_button:
            with st.spinner("Analyzing scene..."):
                description = get_image_description(image_bytes, vision_model)
                st.markdown('<div class="output-text">' + description + '</div>', unsafe_allow_html=True)
                audio_base64 = text_to_speech(description)
                if audio_base64:
                    st.audio(f"data:audio/mp3;base64,{audio_base64}", format="audio/mp3")
        
        if text_button:
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
                st.markdown('<div class="output-text">' + extracted_text + '</div>', unsafe_allow_html=True)
                text_audio_base64 = text_to_speech(extracted_text)
                if text_audio_base64:
                    st.audio(f"data:audio/mp3;base64,{text_audio_base64}", format="audio/mp3")
        
        if objects_button:
            with st.spinner("Detecting objects..."):
                objects = detect_objects(image_bytes, vision_model)
                st.markdown('<div class="output-text">' + objects + '</div>', unsafe_allow_html=True)
                objects_audio_base64 = text_to_speech(objects)
                if objects_audio_base64:
                    st.audio(f"data:audio/mp3;base64,{objects_audio_base64}", format="audio/mp3")
        
        if guidance_button:
            with st.spinner("Generating guidance..."):
                description = get_image_description(image_bytes, vision_model)
                guidance = get_task_specific_guidance(description, text_model)
                st.markdown('<div class="output-text">' + guidance + '</div>', unsafe_allow_html=True)
                guidance_audio_base64 = text_to_speech(guidance)
                if guidance_audio_base64:
                    st.audio(f"data:audio/mp3;base64,{guidance_audio_base64}", format="audio/mp3")

if __name__ == "__main__":
    main()
