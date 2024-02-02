# First Stage: Build the annotation interface
FROM node:20 as build-interface

WORKDIR /app

# Copy package.json and package-lock.json (if available) first
COPY annotation-interface/package*.json /app/
RUN npm install

# Now copy the rest of the files
COPY annotation-interface /app
RUN npm run build-under-base

# Second Stage: Setup and Run the Python server
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
  apt-get install -y \
  tree \
  curl \
  git \
  python3-opencv

WORKDIR /app

# Copying the built interface files
COPY --from=build-interface /app/dist /app/public

# Copying the Python server
COPY annotation-server /app/

RUN curl -o sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Install Python dependencies
COPY requirements.txt /app

RUN pip install -r requirements.txt

EXPOSE 1734

CMD tree && python3 /app/main.py
