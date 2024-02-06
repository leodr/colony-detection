# First Stage: Build the annotation interface
FROM node:20 as interface-builder

WORKDIR /app

# Copy package.json and package-lock.json (if available) first
COPY interface/package*.json /app/
RUN npm install

# Now copy the rest of the files
COPY interface /app
RUN npm run build-under-base

# Second Stage: Setup and Run the Python server
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime as main

# Set environment variables (important for avoiding opencv installation prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV IS_DOCKER_CONTAINER=1

# Install system dependencies
RUN apt-get update && \
  apt-get install -y \
  curl \
  git \
  python3-opencv

WORKDIR /app

# Download the SAM model file
RUN curl -o sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy the built interface files
COPY --from=interface-builder /app/dist ./interface/dist

# Copy the Python server
COPY . .

EXPOSE 80

CMD ["python3", "main.py"] 
