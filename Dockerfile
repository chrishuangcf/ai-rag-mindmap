# Use the common base image
FROM langchain-rag-base:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --upgrade pip && \
    pip install --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set application-specific environment variables
ENV LANGSMITH_TRACING=true
ENV LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
ENV LANGSMITH_PROJECT="my_langchain_project"
ENV LANGSMITH_API_KEY=<<key>>
ENV OPENAI_API_KEY=<<key>>
ENV LLM_PROVIDER="openrouter"

# Expose the port the app runs on (e.g., 8000 for FastAPI)
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

