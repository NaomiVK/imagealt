import streamlit as st
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import csv
import streamlit.components.v1 as components
import os
from groq import Groq
import json


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
    # Retrieve your Groq API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: No Groq API key found."

    client = Groq(api_key=api_key)

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
            "You are an advanced assistant that analyzes images. "
            "Look at the image carefully (in base64) and provide a factual, concise description "
            "of the main objects, scene, and context you observe. "
            "Do not hallucinate details not present in the image. Keep the description under 20 words, and DO NOT truncate your response"
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
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.7,  # Lower temperature
            max_tokens=max_tokens,
            top_p=0.90,  
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting analysis from Groq: {str(e)}"


def translate_to_french(text: str) -> str:
    """Translate text to French using Groq API"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        return ""

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text from English to French. Provide ONLY the direct translation without any explanations, notes, or additional commentary. Maintain the same tone and style of the original text."},
                {"role": "user", "content": text}
            ],
            model="mixtral-8x7b-32768",
            temperature=0.3,
            max_tokens=1024
        )
        translation = chat_completion.choices[0].message.content.strip()
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


def copy_button(key, text):
    safe_text = f"'{text}'"
    html_str = f'''
    <button 
        onclick="navigator.clipboard.writeText({safe_text}).then(() => {{
            this.innerText='Copied!';
            setTimeout(() => this.innerText='Copy {key}', 2000);
        }})" 
        style="background-color:#4CAF50;border:none;color:white;padding:10px 20px;
               text-align:center;text-decoration:none;display:inline-block;font-size:16px;
               margin:4px 2px;cursor:pointer;border-radius:4px;"
    >
        Copy {key}
    </button>
    '''
    components.html(html_str, height=50)


def main():
    st.title("Image alt text")
    st.write("Upload images or PDFs to create image alt text with French translations")

    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'file_data' not in st.session_state:
        st.session_state.file_data = {}

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
                            # Translate long description
                            with st.spinner(f'Translating description for page {idx + 1}...'):
                                french_long_desc = translate_to_french(long_desc)
                            pdf_data.append({
                                'image': img,
                                'long_desc': long_desc,
                                'french_long_desc': french_long_desc
                            })
                        st.session_state.file_data[file.name] = {'type': 'pdf', 'data': pdf_data}

                    else:
                        image = Image.open(file)
                        image = resize_image_if_needed(image, max_size=1024)
                        analysis = get_vision_analysis(image)
                        # Translate image analysis
                        with st.spinner('Translating description...'):
                            french_analysis = translate_to_french(analysis)
                        st.session_state.file_data[file.name] = {
                            'type': 'image',
                            'data': {
                                'image': image,
                                'analysis': analysis,
                                'french_analysis': french_analysis
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("English Description:")
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("English Alt Text Description:")
                st.write(file_data['data']['analysis'])
                copy_button(f"Alt Text for {filename}", file_data['data']['analysis'])
            
            with col2:
                st.write("Description Alternative Française:")
                st.write(file_data['data']['french_analysis'])
                copy_button(f"Alt Text (FR) for {filename}", file_data['data']['french_analysis'])
            
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
