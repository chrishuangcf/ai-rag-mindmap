# Use the common base image
FROM langchain-rag-base:latest

# Create a non-root user and group
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Create application and upload directories, and set permissions BEFORE copying code
RUN mkdir -p /home/appuser/app/uploads && chown -R appuser:appgroup /home/appuser/app

# Set the working directory
WORKDIR /home/appuser/app

# Copy requirements and install as root, but in the appuser's directory
COPY transcoder/requirements.txt .
RUN pip install --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --upgrade pip && \
    pip install --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy the application code into the directory, ensuring it's owned by the appuser
COPY --chown=appuser:appgroup transcoder/ .

# Switch to the non-root user
USER appuser

# Define the volume
VOLUME /home/appuser/app/uploads

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "server:app"]
