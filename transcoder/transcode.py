"""
Core transcoding logic using PyMuPDF for PDFs and pypandoc for DOCX.
"""
import os
import pypandoc
import fitz  # PyMuPDF

def convert_to_markdown(filepath: str) -> str:
    """
    Converts a file (PDF, DOCX, Markdown) to Markdown text.
    Deletes the file immediately after conversion.

    Args:
        filepath: The absolute path to the uploaded file.

    Returns:
        The Markdown content as a string.
    
    Raises:
        Exception: If the conversion fails or the file cannot be deleted.
    """
    try:
        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == '.pdf':
            # Use PyMuPDF to extract text from PDF
            doc = fitz.open(filepath)
            markdown_text = ""
            for page in doc:
                markdown_text += page.get_text()
            doc.close()
            return markdown_text
        
        elif file_extension == '.docx':
            # Use pypandoc for DOCX conversion
            return pypandoc.convert_file(filepath, 'markdown')
        
        elif file_extension in ['.md', '.markdown']:
            # For Markdown files, read and return content directly
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        # Re-raise any errors to be caught by the server
        raise e
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                print(f"Error: Could not delete temporary file {filepath}: {e}")

