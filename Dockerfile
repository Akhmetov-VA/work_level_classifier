# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY poetry.lock /app/poetry.lock
COPY pyproject.toml /app/pyproject.toml

# Install any needed packages specified in requirements.txt
RUN python -m pip install poetry==1.8.3
RUN poetry install --only main

# Run app.py when the container launches
CMD ["streamlit", "run", "classifier/app.py"]
