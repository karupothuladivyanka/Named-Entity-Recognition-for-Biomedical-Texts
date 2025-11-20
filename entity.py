import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import tempfile
import re
import os
import pandas as pd
import html
from datetime import datetime

# Configure Gemini API
API_KEY = "AIzaSyDot8VOtEx6PFIDTN7JBBuVgg-sznlqiMM"  # Replace with your actual Gemini API key
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

    # Debug toggle
    st.session_state['debug_mode'] = st.sidebar.checkbox("Debug Mode", st.session_state['debug_mode'])

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    text_input = st.text_area("Or paste your biomedical text here:")

    extracted_text = ""
    
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            extracted_text = read_pdf(uploaded_file)
            # Show a sample of the extracted text
            if extracted_text:
                st.sidebar.success(f"Successfully extracted {len(extracted_text)} characters from PDF")
                if st.session_state['debug_mode']:
                    st.sidebar.subheader("PDF Info")
                    st.sidebar.write(st.session_state['debug_pdf_info'])
                    st.sidebar.write(f"Pages: {st.session_state['debug_pdf_pages']}")
                    st.sidebar.write(f"Extracted text length: {st.session_state['debug_extracted_text_length']}")
                    st.sidebar.subheader("Sample of extracted text")
                    st.sidebar.text(extracted_text[:500] + "...")
    else:
        extracted_text = text_input.strip()

    if not extracted_text:
        st.warning("Please upload a PDF or enter some text.")
        return

    if st.button("Analyze Text"):
        with st.spinner("Extracting entities..."):
            entities = extract_entities(extracted_text)
            
            if st.session_state['debug_mode'] and 'debug_response' in st.session_state:
                st.sidebar.subheader("Raw Gemini Response")
                st.sidebar.text(st.session_state['debug_response'])

        if not entities:
            st.error("No entities were extracted. Please check the text and try again.")
            return

        st.subheader("Entities Found:")
        st.dataframe(pd.DataFrame(entities))

        st.subheader("Highlighted Entities in Text:")
        visualize_entities(extracted_text, entities)
        
        if st.session_state['debug_mode']:
            st.sidebar.write(f"Number of entities for highlighting: {st.session_state['debug_entities_count']}")

        with st.spinner("Analyzing sentiment and context..."):
            sentiment_analysis = analyze_sentiment_and_context(entities, extracted_text)

        st.subheader("Sentiment and Context Analysis (Top 8 Diseases):")
        st.write(sentiment_analysis)

        with st.spinner("Generating insights..."):
            insights = generate_general_insights(extracted_text, entities)

        st.subheader("General Insights:")
        st.write(insights)

if __name__ == "__main__":
    main()