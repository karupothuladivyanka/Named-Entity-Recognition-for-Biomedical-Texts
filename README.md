# Named-Entity-Recognition-for-Biomedical-Texts
This project is a Biomedical Named Entity Recognition (NER) web application built using Flask and Google Gemini AI. It extracts important biomedical terms such as diseases, drugs, genes, proteins, chemicals, and symptoms from PDF files, website URLs, or plain text.
How It Works
1. Input Text
You can upload a PDF, enter text manually, or provide a URL.
2. Extract Content
The app extracts readable text from the input source.
3. Check Biomedical Relevance
The system verifies whether the content contains biomedical information.
4. Named Entity Recognition
Gemini AI identifies key biomedical entities (e.g., DISEASE, DRUG, GENE).
5.Entity Filtering
Only the most relevant entities are selected for visualization.
6.Highlight Entities
The text is displayed with color-coded highlights for each entity type.
7. Generate Insights
The app creates general insights based on extracted biomedical entities.
8. Disease Sentiment Analysis
It analyzes the sentiment, importance, and context of major diseases mentioned.
9. Results Displayed
The user sees highlighted text, insights, sentiment analysis, and entity counts.
