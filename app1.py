import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import tempfile
import re
import os
import pandas as pd
import html
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Configure Gemini API
API_KEY = ""  # Replace with your actual Gemini API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Define entity colors for visualization
ENTITY_COLORS = {
    "DISEASE": "#ff9966",    # Orange
    "DRUG": "#8aff80",       # Light Green
    "DRUG_CLASS": "#8aff80", # Light Green (same as DRUG)
    "DOSAGE": "#ff6b6b",     # Red
    "FORM": "#f0e68c",       # Khaki
    "FREQUENCY": "#ffa500",  # Orange
    "DURATION": "#ffff00",   # Yellow
    "ROUTE": "#add8e6",      # Light Blue
    "REASON": "#98fb98",     # Pale Green
    "SYMPTOM": "#d8bfd8",    # Thistle
    "ORGAN": "#afeeee",      # Pale Turquoise
    "PROTEIN": "#87cefa",    # Light Sky Blue
    "GENE": "#dda0dd",       # Plum
    "CHEMICAL": "#b0c4de",   # Light Steel Blue
    "ORGANIZATION": "#f5deb3", # Wheat
    "LOCATION": "#d3d3d3",   # Light Gray
    "VIRUS": "#ffcccb",      # Light Red
    "HORMONE": "#98fb98"     # Pale Green
}

def extract_entities(text):
    """Extract unique entities using Gemini API"""
    if not text or not isinstance(text, str):
        return []
    try:
        prompt = f"""
Extract named entities from the following biomedical text. 
Provide the output in the following format exactly:
Entity - [Label]
Text: {text[:4000]}
Rules:
1. Identify entities such as diseases, chemicals, genes, proteins, drugs, dosages, frequency, duration, form, viruses, and hormones.
2. Assign appropriate labels (e.g., DISEASE, CHEMICAL, GENE, PROTEIN, DRUG, DRUG_CLASS, DOSAGE, FREQUENCY, DURATION, FORM, VIRUS, HORMONE).
3. Be concise and specific.
4. For drug classes, use DRUG_CLASS label.
5. Make sure every entity in the text is identified.
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                extracted_text = response.text.strip()
                st.session_state['debug_response'] = extracted_text  # Save for debugging
                entities = parse_gemini_response(extracted_text)
                # Remove duplicates by converting to a set of tuples
                unique_entities = list({(e["entity"], e["label"]) for e in entities})
                # Convert back to list of dictionaries
                return [{"entity": e[0], "label": e[1]} for e in unique_entities]
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error extracting entities: {str(e)}")
                    return []
                continue
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return []

def check_biomedical_content(text):
    """Check if the text contains biomedical content"""
    if not text or not isinstance(text, str):
        return False
    
    prompt = f"""
Determine if the following text contains biomedical content.
Answer with only YES or NO.
Biomedical content includes medical terminology, disease names, drug names, 
treatment protocols, clinical trials, medical research, etc.
Text: {text[:3000]}
"""
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        return "YES" in answer
    except Exception as e:
        st.error(f"Error checking biomedical content: {str(e)}")
        return False

def parse_gemini_response(response):
    """Parse Gemini's response into entities"""
    entities = []
    entity_pattern = re.compile(r"(.*?)\s*-\s*\[(.*?)\]")
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        entity_match = entity_pattern.match(line)
        if entity_match:
            entity = entity_match.group(1).strip()
            label = entity_match.group(2).strip().upper()
            if entity and label:
                entities.append({"entity": entity, "label": label})
    return entities

def read_pdf(file):
    """Extract text from a PDF file"""
    if not file:
        st.error("No file provided")
        return ""
    try:
        # For debugging, show the PDF file info
        st.session_state['debug_pdf_info'] = f"File type: {type(file)}, Name: {file.name}, Size: {file.size} bytes"
        
        # Create a temporary file to ensure PyPDF2 can read it properly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Read the PDF
        pdf_reader = PdfReader(tmp_file_path)
        st.session_state['debug_pdf_pages'] = len(pdf_reader.pages)
        
        full_text = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)
            else:
                st.warning(f"No text extracted from page {page_num+1}")
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        extracted_text = "\n".join(full_text)
        st.session_state['debug_extracted_text_length'] = len(extracted_text)
        
        if not extracted_text:
            st.warning("No text could be extracted from the PDF. The file might be scanned images or protected.")
        
        return extracted_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_url_content(url):
    """Extract content from a URL"""
    if not url:
        st.error("No URL provided")
        return ""
    
    try:
        # Check if URL is valid and add schema if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse URL to ensure it's valid
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.netloc:
            st.error("Invalid URL format")
            return ""
        
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Failed to retrieve content. Status code: {response.status_code}")
            return ""
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        st.session_state['debug_extracted_text_length'] = len(text)
        st.session_state['debug_url_source'] = url
        
        if not text:
            st.warning("No text could be extracted from the URL.")
        
        return text
    except Exception as e:
        st.error(f"Error extracting content from URL: {str(e)}")
        return ""

def analyze_sentiment_and_context(entities, text):
    """Analyze sentiment and context of top 8 diseases"""
    # Filter entities to get only diseases
    disease_entities = [e["entity"] for e in entities if e["label"] == "DISEASE"]
    if not disease_entities:
        return "No diseases found for sentiment analysis."
    
    # Limit to top 8 diseases
    top_diseases = disease_entities[:8]
    prompt = f"""
Analyze the sentiment and contextual importance of these diseases in the following text.
For each disease, determine:
1. Sentiment (positive, negative, or neutral)
2. Confidence level (high, medium, low)
3. Contextual importance (critical, important, or peripheral)
4. Brief justification for the assessment
Diseases: {', '.join(top_diseases)}
Text: {text[:3000]}
"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        return response_text
    except Exception as e:
        st.warning(f"Error analyzing sentiment: {str(e)}")
        return "Error analyzing sentiment."

def find_entity_positions(text, entity_text):
    """Find all positions of an entity in the text"""
    positions = []
    start_idx = 0
    text_lower = text.lower()
    entity_lower = entity_text.lower()
    
    # Handle empty entities
    if not entity_text or not entity_lower:
        return positions
        
    while True:
        start_pos = text_lower.find(entity_lower, start_idx)
        if start_pos == -1:
            break
        actual_entity = text[start_pos:start_pos + len(entity_text)]
        end_pos = start_pos + len(entity_text)
        positions.append((start_pos, end_pos, actual_entity))
        start_idx = start_pos + 1
    return positions

def create_html_with_highlights(text, entities):
    """Create HTML with highlighted entities"""
    if not text or not entities:
        return "No text or entities to visualize"
    
    # First, escape the entire text
    safe_text = html.escape(text)
    
    # Find all entity positions
    all_positions = []
    for item in entities:
        entity = item["entity"]
        label = item["label"]
        color = ENTITY_COLORS.get(label, "#cccccc")
        positions = find_entity_positions(text, entity)
        for start, end, actual_text in positions:
            all_positions.append((start, end, actual_text, label, color))
    
    # Sort by start position
    all_positions.sort(key=lambda x: x[0])
    
    # Remove overlapping entities
    non_overlapping = []
    last_end = -1
    for pos in all_positions:
        start, end, actual_text, label, color = pos
        if start >= last_end:
            non_overlapping.append(pos)
            last_end = end
    
    # Build the HTML
    result_html = []
    last_end = 0
    for start, end, actual_text, label, color in non_overlapping:
        if start > last_end:
            result_html.append(safe_text[last_end:start])
        result_html.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{label}">{html.escape(actual_text)}</span>')
        last_end = end
    
    if last_end < len(safe_text):
        result_html.append(safe_text[last_end:])
    
    return "".join(result_html)

def visualize_entities(text, entities):
    """Display text with highlighted entities"""
    if not text or not entities:
        st.markdown("No entities found in the text for visualization")
        return
    
    # For debugging, show how many entities we're trying to highlight
    st.session_state['debug_entities_count'] = len(entities)
    
    highlighted_html = create_html_with_highlights(text, entities)
    
    # For very long texts, we might need to chunk the display
    if len(highlighted_html) > 100000:
        st.warning("The text is very long. Displaying first 100,000 characters only.")
        highlighted_html = highlighted_html[:100000] + "... (truncated)"
    
    st.markdown(
        f'<div style="white-space: pre-wrap; font-family: sans-serif; border: 1px solid #ddd; padding: 10px; border-radius: 5px; max-height: 500px; overflow-y: auto;">{highlighted_html}</div>',
        unsafe_allow_html=True
    )

def generate_general_insights(text, entities):
    """Generate general insights based on entities and text"""
    if not text or not entities:
        return "No insights could be generated due to missing data."
    entity_types = set(e["label"] for e in entities)
    entity_names = [e["entity"] for e in entities]
    prompt = f"""
Analyze this biomedical text and provide general insights about the key entities and their significance.
Text summary: {text[:1000]}
Entity types present: {', '.join(entity_types)}
Key entities: {', '.join(entity_names[:10])}
"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        return response_text
    except Exception as e:
        st.warning(f"Error generating insights: {str(e)}")
        return "Error generating insights."

def main():
    st.title("Biomedical Text Analysis Tool")
    
    # Initialize session state for debugging
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
        st.session_state['debug_pdf_info'] = ""
        st.session_state['debug_pdf_pages'] = 0
        st.session_state['debug_extracted_text_length'] = 0
        st.session_state['debug_entities_count'] = 0
        st.session_state['debug_response'] = ""
        st.session_state['debug_url_source'] = ""

    # Debug toggle
    st.session_state['debug_mode'] = st.sidebar.checkbox("Debug Mode", st.session_state['debug_mode'])

    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Upload PDF", "Enter Text", "Enter URL"])
    
    extracted_text = ""
    
    with tab1:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file:
            with st.spinner("Reading PDF..."):
                extracted_text = read_pdf(uploaded_file)
                # Show a sample of the extracted text
                if extracted_text:
                    st.sidebar.success(f"Successfully extracted {len(extracted_text)} characters from PDF")
                    if st.session_state['debug_mode']:
                        st.sidebar.subheader("PDF Info")
                        st.sidebar.write(st.session_state['debug_pdf_info'])
                        st.sidebar.write(f"Number of pages: {st.session_state['debug_pdf_pages']}")
                        st.sidebar.write(f"Extracted text length: {st.session_state['debug_extracted_text_length']} characters")
                    # Display a preview of the extracted text
                    st.subheader("Extracted Text Preview")
                    st.text_area("Preview", value=extracted_text[:1000], height=200)
    with tab2:
        input_text = st.text_area("Enter biomedical text here", height=300)
        if input_text:
            extracted_text = input_text
            if st.session_state['debug_mode']:
                st.sidebar.subheader("Input Text Info")
                st.sidebar.write(f"Text length: {len(extracted_text)} characters")
    with tab3:
        url_input = st.text_input("Enter a URL to extract content")
        if url_input:
            with st.spinner("Extracting content from URL..."):
                extracted_text = extract_url_content(url_input)
                if extracted_text:
                    st.sidebar.success(f"Successfully extracted {len(extracted_text)} characters from URL")
                    if st.session_state['debug_mode']:
                        st.sidebar.subheader("URL Source Info")
                        st.sidebar.write(f"Source URL: {st.session_state['debug_url_source']}")
                        st.sidebar.write(f"Extracted text length: {st.session_state['debug_extracted_text_length']} characters")
                    # Display a preview of the extracted text
                    st.subheader("Extracted Text Preview")
                    st.text_area("Preview", value=extracted_text[:1000], height=200)

    # Check if there's any text to analyze
    if not extracted_text:
        st.info("Please upload a PDF, enter text, or provide a URL to proceed.")
        return

    # Check if the text contains biomedical content
    with st.spinner("Checking for biomedical content..."):
        is_biomedical = check_biomedical_content(extracted_text)
        if not is_biomedical:
            st.warning("The provided text does not appear to contain biomedical content.")
            return

    # Extract entities
    with st.spinner("Extracting named entities..."):
        entities = extract_entities(extracted_text)
        if not entities:
            st.warning("No entities were found in the text.")
            return

    # Visualize entities
    st.subheader("Named Entity Recognition (NER) Visualization")
    visualize_entities(extracted_text, entities)

    # Generate general insights
    with st.spinner("Generating general insights..."):
        insights = generate_general_insights(extracted_text, entities)
        st.subheader("General Insights")
        st.markdown(insights)

    # Analyze sentiment and context for diseases
    with st.spinner("Analyzing sentiment and context for diseases..."):
        sentiment_analysis = analyze_sentiment_and_context(entities, extracted_text)
        st.subheader("Disease Sentiment and Context Analysis")
        st.markdown(sentiment_analysis)

if __name__ == "__main__":
    main()