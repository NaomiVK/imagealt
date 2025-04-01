import streamlit as st
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import csv
import streamlit.components.v1 as components
import os
import json
import requests
from langdetect import detect


def convert_pdf_to_images(pdf_bytes, max_image_size=1024):
    try:
        images = convert_from_bytes(pdf_bytes)
        resized_images = []
        for img in images:
            img = resize_image_if_needed(img, max_image_size)
            resized_images.append(img)
        return resized_images
    except Exception as e:
        st.error(f"Error converting PDF: {str(e)}")
        return []


def resize_image_if_needed(image, max_size):
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image


def get_vision_analysis(image, is_pdf=False):
    # Get the selected vision model from session state
    vision_model = st.session_state.vision_model

    # Convert the image to base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    base64_size_bytes = len(img_base64)
    base64_size_kb = base64_size_bytes / 1024

    st.write(f"Image Base64 size: {base64_size_kb:.2f} KB")  # Debugging

    # Construct prompt & tokens
    if is_pdf:
        prompt = "Provide a thorough description of the text content in this page. Be concise and don't truncate your response"
        max_tokens = 500
    else:
        prompt = (
            "Create a short, concise alt text for this image suitable for a website. "
            "DO NOT start with phrases like 'The image depicts', 'The image shows', or similar. "
            "Instead, directly describe the main subject in 15-20 words maximum. "
            "Focus only on the key elements necessary for accessibility. "
            "Use simple, direct language without unnecessary words."
        )
        max_tokens = 30  # Give a few more tokens to avoid truncation, while setting expectation

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Use OpenRouter for vision analysis
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: No OpenRouter API key found."

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            "model": vision_model,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "top_p": 0.85
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error getting analysis from OpenRouter: {str(e)}"


def detect_language(text: str) -> str:
    """Detect the language of the given text.
    
    Args:
        text: The text to detect language for
        
    Returns:
        The language code (e.g., 'en', 'fr')
    """
    try:
        return detect(text)
    except Exception as e:
        st.error(f"Language detection error: {str(e)}")
        return "en"  # Default to English if detection fails


def translate_to_french(text: str) -> str:
    """Translate text to French using OpenRouter API with mistralai/mixtral-8x7b-instruct model.
    
    Skip translation if the text is already in French.
    """
    # First detect the language
    lang = detect_language(text)
    
    # If already French, return the original text
    if lang == 'fr':
        return text
        
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        st.error("OPENROUTER_API_KEY not found in environment variables")
        return ""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a professional translator. Translate the following text from English to French. Provide ONLY the direct translation without any explanations, notes, or additional commentary. Maintain the same tone and style of the original text."},
                {"role": "user", "content": text}
            ],
            "model": "mistralai/mixtral-8x7b-instruct",
            "temperature": 0.3,
            "max_tokens": 1024,
            "top_p": 0.85
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        translation = response.json()["choices"][0]["message"]["content"].strip()
        return translation
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""

def write_to_csv(image_name: str, description: str = None, long_description: str = None,
                french_description: str = None, french_long_description: str = None):
    """Write descriptions to CSV file with French translations"""
    file_exists = False
    try:
        with open('image_descriptions.csv', 'r', encoding='utf-8') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open('image_descriptions.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Image filename",
                "Image alt text (English)",
                "Image alt text (French)",
                "PDF description (English)",
                "PDF description (French)"
            ])
        
        if long_description is not None:
            writer.writerow([
                image_name,
                description or "",
                french_description or "",
                long_description,
                french_long_description or ""
            ])
        else:
            writer.writerow([
                image_name,
                description or "",
                french_description or "",
                "",
                ""
            ])


def get_csv_download_link():
    try:
        with open('image_descriptions.csv', 'r', encoding='utf-8') as f:
            csv_content = f.read()
    except FileNotFoundError:
        return "No CSV file to download yet."

    b64 = base64.b64encode(csv_content.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="image_descriptions.csv">Download CSV</a>'


def copy_button(text):
    """Create a copy button using HTML/JavaScript via components.html"""
    button_id = f"copy_btn_{hash(text)}"
    html_str = f"""
        <button id="{button_id}" onclick="copyText(this, `{text}`)" 
            style="background-color:#4CAF50;border:none;color:white;padding:10px 20px;text-align:center;
            text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:4px;">
            Copy Text
        </button>
        <script>
        function copyText(btn, text) {{
            navigator.clipboard.writeText(text).then(() => {{
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy Text', 2000);
            }});
        }}
        </script>
    """
    components.html(html_str, height=50)


def reset_app_state():
    """Reset the app state when model changes"""
    st.session_state.processed_files = []
    st.session_state.file_data = {}

def main():
    # Title and description at the top of the page
    st.title("Image alt text")
    st.write("Upload images or PDFs to create image alt text with French translations")
    
    # Initialize session state
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'file_data' not in st.session_state:
        st.session_state.file_data = {}

    # Handle model migration from Grok to GPT-4 Vision
    if 'vision_model' not in st.session_state or st.session_state.vision_model in ["x-ai/grok-2-vision-1212", "meta-llama/llama-3.2-11b-vision-instruct"]:
        st.session_state.vision_model = "openai/gpt-4-vision-preview"

    if 'previous_model' not in st.session_state:
        st.session_state.previous_model = st.session_state.vision_model
    
    # Add option to choose vision model
    vision_model = st.selectbox(
        "Choose Vision Model:",
        [
            "openai/gpt-4-vision-preview",
            "google/gemini-pro-vision",
            "meta-llama/llama-3.2-11b-vision-instruct"
        ],
        index=[
            "openai/gpt-4-vision-preview",
            "google/gemini-pro-vision",
            "meta-llama/llama-3.2-11b-vision-instruct"
        ].index(st.session_state.vision_model)
    )
    
    # Check if model has changed
    if vision_model != st.session_state.previous_model:
        reset_app_state()
        st.session_state.vision_model = vision_model
        st.session_state.previous_model = vision_model
        st.info(f"Changed model to {vision_model}. App has been reset.")
    else:
        st.session_state.vision_model = vision_model

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            progress_bar = st.progress(0)
            status_container = st.empty()

            total_files = len(new_files)
            for i, file in enumerate(new_files):
                with st.spinner(f'Processing file {i + 1} of {total_files}: {file.name}'):
                    st.write(f"Processing: {file.name}")

                    if file.type == "application/pdf":
                        pdf_images = convert_pdf_to_images(file.read())
                        st.write(f"Found {len(pdf_images)} pages in PDF")

                        pdf_data = []
                        for idx, img in enumerate(pdf_images):
                            status_container.info(f"Processing page {idx + 1} of {len(pdf_images)} for {file.name}")
                            long_desc = get_vision_analysis(img, is_pdf=True)
                            
                            # Detect language of the content
                            lang = detect_language(long_desc)
                            st.write(f"Detected language: {lang}")
                            
                            # Translate long description (will skip if already French)
                            with st.spinner(f'Translating description for page {idx + 1}...'):
                                french_long_desc = translate_to_french(long_desc)
                            
                            pdf_data.append({
                                'image': img,
                                'long_desc': long_desc,
                                'french_long_desc': french_long_desc,
                                'detected_language': lang
                            })
                        st.session_state.file_data[file.name] = {'type': 'pdf', 'data': pdf_data}

                    else:
                        image = Image.open(file)
                        image = resize_image_if_needed(image, max_size=1024)
                        analysis = get_vision_analysis(image)
                        
                        # Detect language of the content
                        lang = detect_language(analysis)
                        st.write(f"Detected language: {lang}")
                        
                        # Translate image analysis (will skip if already French)
                        with st.spinner('Translating description...'):
                            french_analysis = translate_to_french(analysis)
                            
                        st.session_state.file_data[file.name] = {
                            'type': 'image',
                            'data': {
                                'image': image,
                                'analysis': analysis,
                                'french_analysis': french_analysis,
                                'detected_language': lang
                            }
                        }

                    st.session_state.processed_files.append(file.name)
                    progress_bar.progress((i + 1) / total_files)

            st.success("All files processed and translated successfully!")

    # Display results
    for filename in st.session_state.processed_files:
        file_data = st.session_state.file_data[filename]
        if file_data['type'] == 'pdf':
            st.subheader(f"PDF: {filename}")
            for idx, page_data in enumerate(file_data['data']):
                st.write(f"Page {idx + 1}:")
                st.image(page_data['image'], caption=f"Page {idx + 1} from PDF")
                
                # Display detected language
                st.write(f"Detected language: {page_data.get('detected_language', 'unknown')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Description:")
                    st.write(page_data['long_desc'])
                
                with col2:
                    st.write("Description Française:")
                    st.write(page_data['french_long_desc'])
                
                write_to_csv(
                    f"{filename}_page_{idx + 1}",
                    None,
                    page_data['long_desc'],
                    None,
                    page_data['french_long_desc']
                )
                st.divider()
        else:
            st.subheader(f"Image: {filename}")
            st.image(file_data['data']['image'], caption="Uploaded Image")
            
            # Display detected language
            st.write(f"Detected language: {file_data['data'].get('detected_language', 'unknown')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Alt Text Description:")
                st.write(file_data['data']['analysis'])
                copy_button(file_data['data']['analysis'])
            
            with col2:
                st.write("Description Alternative Française:")
                st.write(file_data['data']['french_analysis'])
                copy_button(file_data['data']['french_analysis'])
            
            write_to_csv(
                filename,
                file_data['data']['analysis'],
                None,
                file_data['data']['french_analysis']
            )
            st.divider()

    if st.session_state.processed_files:
        st.markdown(get_csv_download_link(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
