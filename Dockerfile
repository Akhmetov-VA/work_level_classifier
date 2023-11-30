# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r requirements.txt

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
