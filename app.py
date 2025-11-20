import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import tempfile
import re
import os
import pandas as pd
import random
import html

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

def extract_entities_and_relationships(text):
    """Extract entities and relationships using Gemini API"""
    if not text or not isinstance(text, str):
        return [], []
    try:
        # Define a prompt for entity and relationship extraction
        prompt = f"""
Extract named entities and their relationships from the following biomedical text. 
Provide the output in the following format exactly:

Entity - [Label]
Relationship: Entity1 -[relationship]-> Entity2

Text: {text[:4000]}

Rules:
1. Identify entities such as diseases, chemicals, genes, proteins, drugs, dosages, frequency, duration, form, viruses, and hormones.
2. Assign appropriate labels (e.g., DISEASE, CHEMICAL, GENE, PROTEIN, DRUG, DRUG_CLASS, DOSAGE, FREQUENCY, DURATION, FORM, VIRUS, HORMONE).
3. Identify meaningful relationships between entities (e.g., "treats", "causes", "regulates").
4. Be concise and specific.
5. For drug classes, use DRUG_CLASS label.
6. Make sure every entity in the text is identified.
"""
        # Generate response
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                extracted_text = response.text.strip()
                # Parse the response into entities and relationships
                entities, relationships = parse_gemini_response(extracted_text)
                return entities, relationships
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Error extracting entities: {str(e)}")
                    return [], []
                continue
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return [], []

def parse_gemini_response(response):
    """Parse Gemini's response into entities and relationships"""
    entities = []
    relationships = []
    entity_pattern = re.compile(r"(.*?)\s*-\s*\[(.*?)\]")
    relationship_pattern = re.compile(r"Relationship:\s*(.*?)\s*-\[(.*?)\]->\s*(.*)")
    alt_relationship_pattern = re.compile(r"(.*?)\s*-\[(.*?)\]->\s*(.*)")
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to match entity pattern
        entity_match = entity_pattern.match(line)
        if entity_match:
            entity = entity_match.group(1).strip()
            label = entity_match.group(2).strip().upper()
            if entity and label:
                entities.append({"entity": entity, "label": label})
            continue
            
        # Try to match relationship patterns
        rel_match = relationship_pattern.match(line)
        if not rel_match:
            rel_match = alt_relationship_pattern.match(line)
            
        if rel_match:
            entity1 = rel_match.group(1).strip()
            relation = rel_match.group(2).strip()
            entity2 = rel_match.group(3).strip()
            if entity1 and relation and entity2:
                relationships.append({"entity1": entity1, "relation": relation, "entity2": entity2})
                
    return entities, relationships

def read_pdf(file):
    """Extract text from a PDF file"""
    if not file:
        st.error("No file provided")
        return ""
    try:
        pdf_reader = PdfReader(file)
        full_text = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:  # Only append non-empty text
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def find_entity_positions(text, entity_text):
    """Find all positions of an entity in the text"""
    positions = []
    start_idx = 0
    
    # Convert to lowercase for case-insensitive search
    text_lower = text.lower()
    entity_lower = entity_text.lower()
    
    while True:
        start_pos = text_lower.find(entity_lower, start_idx)
        if start_pos == -1:
            break
            
        # Get the actual case from the original text
        actual_entity = text[start_pos:start_pos + len(entity_text)]
        end_pos = start_pos + len(entity_text)
        
        positions.append((start_pos, end_pos, actual_entity))
        start_idx = start_pos + 1
        
    return positions

def create_html_with_highlights(text, entities):
    """Create HTML with highlighted entities"""
    if not text or not entities:
        return "No text or entities to visualize"
    
    # Convert text to HTML-safe
    safe_text = html.escape(text)
    
    # Create a list of all entity positions
    all_positions = []
    for item in entities:
        entity = item["entity"]
        label = item["label"]
        color = ENTITY_COLORS.get(label, "#cccccc")
        
        positions = find_entity_positions(text, entity)
        for start, end, actual_text in positions:
            all_positions.append((start, end, actual_text, label, color))
    
    # Sort positions by start index
    all_positions.sort(key=lambda x: x[0])
    
    # Handle overlapping entities by keeping only non-overlapping ones
    non_overlapping = []
    last_end = -1
    
    for pos in all_positions:
        start, end, actual_text, label, color = pos
        if start >= last_end:  # No overlap
            non_overlapping.append(pos)
            last_end = end
    
    # Build the HTML with highlights
    result_html = []
    last_end = 0
    
    for start, end, actual_text, label, color in non_overlapping:
        # Add text before the current entity
        if start > last_end:
            result_html.append(safe_text[last_end:start])
        
        # Add the highlighted entity
        result_html.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{label}">{html.escape(actual_text)}</span>')
        
        last_end = end
    
    # Add any remaining text
    if last_end < len(safe_text):
        result_html.append(safe_text[last_end:])
    
    return "".join(result_html)

def visualize_entities(text, entities):
    """Display text with highlighted entities"""
    if not text or not entities:
        st.markdown("No entities found in the text for visualization")
        return
    
    # Generate HTML with highlights
    highlighted_html = create_html_with_highlights(text, entities)
    
    # Display the highlighted text
    st.markdown(
        f'<div style="white-space: pre-wrap; font-family: sans-serif; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">{highlighted_html}</div>',
        unsafe_allow_html=True
    )
    
    # Display entity legend
    st.markdown("### Entity Legend")
    
    # Get unique labels from entities
    unique_labels = sorted(set(item["label"] for item in entities))
    
    # Create a grid layout for the legend
    cols = st.columns(3)
    for i, label in enumerate(unique_labels):
        color = ENTITY_COLORS.get(label, "#cccccc")
        cols[i % 3].markdown(
            f'<div style="display: flex; align-items: center; margin-bottom: 8px;">'
            f'<span style="background-color: {color}; width: 20px; height: 20px; display: inline-block; '
            f'margin-right: 8px; border-radius: 3px;"></span>{label}</div>',
            unsafe_allow_html=True
        )

# Streamlit App
st.set_page_config(page_title="Biomedical NER", layout="wide")
st.title("Biomedical Named Entity Recognition (NER) with Gemini API")

# Input method selection
input_method = st.radio("Select Input Method", ["Text Input", "Upload PDF"])

# Initialize variables
text = ""
entities = []
relationships = []

if input_method == "Text Input":
    # Text input
    text = st.text_area("Enter Biomedical Text", height=200)
    process_button = st.button("Process Text")
    
    if process_button and text:
        with st.spinner("Extracting entities and relationships..."):
            entities, relationships = extract_entities_and_relationships(text)

elif input_method == "Upload PDF":
    # PDF upload
    uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])
    process_pdf_button = st.button("Process PDF")
    
    if process_pdf_button and uploaded_file:
        with st.spinner("Reading PDF and extracting entities..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            text = read_pdf(tmp_file_path)
            os.unlink(tmp_file_path)  # Clean up temporary file
            if text:
                entities, relationships = extract_entities_and_relationships(text)

# Display text with highlighted entities
if text and entities:
    st.header("Text with Highlighted Entities")
    visualize_entities(text, entities)

# Display extracted entities
if entities:
    st.header("Extracted Entities")
    
    # Convert to DataFrame and add index
    entity_df = pd.DataFrame(entities)
    entity_df.insert(0, "", range(len(entity_df)))
    
    # Format entities with asterisk prefix like in the screenshot
    entity_df["entity"] = "* " + entity_df["entity"]
    
    st.dataframe(entity_df, use_container_width=True, hide_index=True)
else:
    if text:  # Only show this message if text was provided
        st.info("No entities extracted. Please provide more detailed biomedical text.")

# Display extracted relationships
if relationships:
    st.header("Extracted Relationships")
    relationship_df = pd.DataFrame(relationships)
    st.dataframe(relationship_df, use_container_width=True)
else:
    if text:  # Only show this message if text was provided
        st.info("No relationships extracted. Please provide text with clear entity relationships.")

# Add requirements information
with st.expander("Installation Requirements"):
    st.markdown("""
    To run this application, you'll need to install the following packages:
    ```
    pip install streamlit google-generativeai PyPDF2 pandas
    ```
    """)