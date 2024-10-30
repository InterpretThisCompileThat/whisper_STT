FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache if they haven't changed
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

# Install Waitress
RUN pip install waitress

# Install necessary packages for selenium
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    wget \
    gnupg \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y \
    google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Create uploads directory
RUN mkdir /app/uploads

# Command to run
CMD ["waitress-serve", "--call", "app:create_app"]
