"""
Flask server for the transcoder service.
Provides an endpoint to upload a file or provide a URL to get back Markdown.
"""
import os
import requests
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from transcode import convert_to_markdown
from urllib.parse import urlparse

# Configuration
UPLOAD_FOLDER = '/home/appuser/app/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'md', 'markdown'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_filename_from_url(url):
    """Extracts a filename from a URL."""
    path = urlparse(url).path
    return os.path.basename(path)

def is_markdown_url(url):
    """Check if URL points to a Markdown file based on extension or path."""
    path = urlparse(url).path.lower()
    return path.endswith('.md') or path.endswith('.markdown')

def fetch_markdown_from_url(url):
    """Fetch Markdown content directly from a URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Try to decode as UTF-8, fallback to other encodings if needed
        try:
            return response.content.decode('utf-8')
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return response.content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            # If all else fails, use utf-8 with error handling
            return response.content.decode('utf-8', errors='replace')
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch Markdown from URL: {e}")

@app.route('/transcode', methods=['POST'])
def transcode_file():
    """
    Handle file transcoding from either a direct file upload or a URL.
    """
    # Case 1: Direct file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                markdown_content = convert_to_markdown(filepath)
                return jsonify({"markdown_content": markdown_content}), 200
            except Exception as e:
                return jsonify({"error": f"Failed to transcode file: {e}"}), 500
        else:
            return jsonify({"error": "File type not allowed. Please upload a PDF, DOCX, or Markdown file."}), 400

    # Case 2: URL input from JSON
    elif request.is_json:
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({"error": "No 'url' provided in JSON payload"}), 400

        try:
            # Special handling for Markdown URLs - fetch directly without file download
            if is_markdown_url(url):
                markdown_content = fetch_markdown_from_url(url)
                return jsonify({
                    "markdown_content": markdown_content,
                    "source_url": url,
                    "file_type": "markdown",
                    "processed_directly": True
                }), 200
            
            # For other file types, download and convert
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status() # Raise an exception for bad status codes

            filename = get_filename_from_url(url)
            if not allowed_file(filename):
                return jsonify({"error": f"File type from URL not allowed: {filename}. Supported: PDF, DOCX, Markdown"}), 400
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert the downloaded file
            markdown_content = convert_to_markdown(filepath)
            return jsonify({
                "markdown_content": markdown_content,
                "source_url": url,
                "file_type": os.path.splitext(filename)[1].lower().lstrip('.'),
                "processed_directly": False
            }), 200

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to download file from URL: {e}"}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to transcode file from URL: {e}"}), 500
    
    # If neither file nor JSON is provided
    else:
        return jsonify({"error": "Invalid request. Please provide a file upload or a JSON payload with a 'url'."}), 400

@app.route('/transcode/markdown', methods=['POST'])
def transcode_markdown_url():
    """
    Handle Markdown file transcoding directly from URL without file download.
    Optimized for Markdown files that don't need conversion.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "No 'url' provided in JSON payload"}), 400
    
    if not is_markdown_url(url):
        return jsonify({"error": "URL does not appear to point to a Markdown file. Use /transcode for other file types."}), 400
    
    try:
        markdown_content = fetch_markdown_from_url(url)
        return jsonify({
            "markdown_content": markdown_content,
            "source_url": url,
            "file_type": "markdown"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint that shows supported file types and service status.
    """
    return jsonify({
        "status": "healthy",
        "service": "transcoder",
        "supported_file_types": {
            "upload": ["pdf", "docx", "md", "markdown"],
            "url": ["pdf", "docx", "md", "markdown"]
        },
        "endpoints": {
            "/transcode": "Upload file or provide URL for transcoding to Markdown",
            "/transcode/markdown": "Optimized endpoint for Markdown URLs (direct fetch)",
            "/health": "Health check and service information"
        },
        "features": {
            "markdown_direct_fetch": True,
            "pdf_support": True,
            "docx_support": True,
            "url_download": True
        }
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
