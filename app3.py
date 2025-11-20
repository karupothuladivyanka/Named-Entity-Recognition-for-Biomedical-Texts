from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from PyPDF2 import PdfReader
import tempfile
import re
import os
import html
import requests
from bs4 import BeautifulSoup
import urllib.parse
import secrets

# Configure Gemini API
API_KEY = "AIzaSyDot8VOtEx6PFIDTN7JBBuVgg-sznlqiMM"  # Replace with your actual Gemini API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

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
                session['debug_response'] = extracted_text  # Save for debugging
                entities = parse_gemini_response(extracted_text)
                # Remove duplicates by converting to a set of tuples
                unique_entities = list({(e["entity"], e["label"]) for e in entities})
                # Convert back to list of dictionaries
                return [{"entity": e[0], "label": e[1]} for e in unique_entities]
            except Exception as e:
                if attempt == max_retries - 1:
                    return []
                continue
    except Exception as e:
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

def read_pdf(file_stream):
    """Extract text from a PDF file"""
    if not file_stream:
        return ""
    try:
        # Create a temporary file to ensure PyPDF2 can read it properly
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_stream.read())
            tmp_file_path = tmp_file.name
        
        # Read the PDF
        pdf_reader = PdfReader(tmp_file_path)
        session['debug_pdf_pages'] = len(pdf_reader.pages)
        
        full_text = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                full_text.append(text)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
        
        extracted_text = "\n".join(full_text)
        session['debug_extracted_text_length'] = len(extracted_text)
        
        return extracted_text
    except Exception as e:
        return ""

def extract_url_content(url):
    """Extract content from a URL"""
    if not url:
        return ""
    
    try:
        # Check if URL is valid and add schema if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse URL to ensure it's valid
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.netloc:
            return ""
        
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
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
        
        session['debug_extracted_text_length'] = len(text)
        session['debug_url_source'] = url
        
        return text
    except Exception as e:
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

def filter_entities_for_visualization(entities):
    """Filter entities to include only the most relevant ones for visualization"""
    if not entities:
        return []
    # Prepare the prompt to filter entities
    entity_list = "\n".join([f"{e['entity']} - [{e['label']}]" for e in entities])
    prompt = f"""
Given the following list of named entities extracted from a biomedical text, 
select only the most relevant entities that should be highlighted in the visualization.remove unnecessary paragraphs with no entity.
Relevant entities are those that are directly related to diseases, drugs, treatments, or significant biological components.
Provide the output in the same format as the input (Entity - [Label]).

Entities:
{entity_list}
"""
    try:
        response = model.generate_content(prompt)
        filtered_text = response.text.strip()
        session['debug_filtered_entities'] = filtered_text  # Save for debugging
        # Parse the filtered response into entities
        filtered_entities = parse_gemini_response(filtered_text)
        return filtered_entities
    except Exception as e:
        return []

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
        return "Error generating insights."

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'pdf-file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf-file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    extracted_text = read_pdf(file)
    if not extracted_text:
        return jsonify({'error': 'Failed to extract text from PDF'}), 400
    
    # Perform analysis
    analysis_result = perform_analysis(extracted_text)
    return jsonify(analysis_result)

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.form
    text = data.get('input-text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform analysis
    analysis_result = perform_analysis(text)
    return jsonify(analysis_result)

@app.route('/analyze-url', methods=['POST'])
def analyze_url():
    data = request.form
    url = data.get('url-input', '')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    extracted_text = extract_url_content(url)
    if not extracted_text:
        return jsonify({'error': 'Failed to extract content from URL'}), 400
    
    # Perform analysis
    analysis_result = perform_analysis(extracted_text)
    return jsonify(analysis_result)

def perform_analysis(text):
    # Check if the text contains biomedical content
    is_biomedical = check_biomedical_content(text)
    if not is_biomedical:
        return {'error': 'The provided text does not appear to contain biomedical content'}
    
    # Extract entities
    entities = extract_entities(text)
    if not entities:
        return {'error': 'No entities were found in the text'}
    
    # Filter entities for visualization
    filtered_entities = filter_entities_for_visualization(entities)
    
    # Create HTML with highlighted entities
    highlighted_html = create_html_with_highlights(text, filtered_entities)
    
    # For very long texts, truncate the display
    if len(highlighted_html) > 100000:
        highlighted_html = highlighted_html[:100000] + "... (truncated)"
    
    # Generate general insights
    insights = generate_general_insights(text, entities)
    
    # Analyze sentiment and context for diseases
    sentiment_analysis = analyze_sentiment_and_context(entities, text)
    
    # Return the analysis results
    # Return the analysis results
    return {
        'success': True,
        'text_preview': text[:1000],
        'highlighted_html': highlighted_html,
        'insights': insights,
        'sentiment_analysis': sentiment_analysis,
        'entities_count': len(filtered_entities),
        'text_length': len(text)
    }

@app.route('/debug', methods=['GET'])
def debug_info():
    """Return debug information from the session"""
    if 'debug_mode' not in request.args or request.args.get('debug_mode') != 'true':
        return jsonify({'error': 'Debug mode not enabled'}), 403
    
    debug_data = {
        'debug_pdf_pages': session.get('debug_pdf_pages', 0),
        'debug_extracted_text_length': session.get('debug_extracted_text_length', 0),
        'debug_entities_count': session.get('debug_entities_count', 0),
        'debug_response': session.get('debug_response', ''),
        'debug_url_source': session.get('debug_url_source', '')
    }
    return jsonify(debug_data)

if __name__ == '__main__':
    app.run(debug=True)