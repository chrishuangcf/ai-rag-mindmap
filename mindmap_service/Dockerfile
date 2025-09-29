# Use the common base image
FROM langchain-rag-base:latest

WORKDIR /app

# Copy application requirements and install Python dependencies
COPY mindmap_service/requirements.txt .
RUN pip install --trusted-host pypi.org     --trusted-host pypi.python.org     --trusted-host files.pythonhosted.org     --upgrade pip &&     pip install --trusted-host pypi.org     --trusted-host pypi.python.org     --trusted-host files.pythonhosted.org     -r requirements.txt

# Download NLTK data with explicit verification
RUN python -m nltk.downloader -d /home/appuser/nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng maxent_ne_chunker words stopwords wordnet && \
    python -c "import nltk; nltk.data.path.append('/home/appuser/nltk_data'); from nltk.tokenize import word_tokenize, sent_tokenize; print('NLTK verification successful')"
    
# Copy application code
COPY mindmap_service/ .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
