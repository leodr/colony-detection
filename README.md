<h1 align="center">
  Bacterial Colony Segmentation
</h1>

<p align="center"><b>
  Project to automatically segment colonies on images of petri dishes.
</b><p>

<br><br>

This repository contains all the building blocks of the colony segmentation
project, including the data annotation app and (eventually) the machine learning
model.

Most of the code is written in Python, excluding the annotation interface, which
is web-based.

<br>

## Setup for Annotation Server

1. **Requirements**

   You need to have Python 3 installed on the system. Install necessary packages
   with

   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyTorch**

   Follow the instructions [here](https://pytorch.org/get-started/locally/) to
   install PyTorch on your system. If you have an NVIDIA GPU available, choose
   the CUDA version.

3. **Download the Segment Anything model file**

   Download the ViT-H SAM model file from
   [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
   If the link is dead, check the
   [repo](https://github.com/facebookresearch/segment-anything#model-checkpoints)
   for the download link.

   Rename the file to `sam_vit_h.pth` and put it into the `annotation-server`
   folder.

   If you have curl, you can use this command to download it and put it into the
   right spot:

   ```bash
   curl -o sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

4. **Run the application**

   Open the terminal inside the `annotation-server` folder and run it with

   ```bash
   python3 main.py
   ```

<br>

## Setup for Annotation Interface

1. **Requirements**

   You need Node.js installed on your system. Choose the LTS release from
   [here](https://nodejs.org/en).

2. **Install Dependencies**

   Install all necessary dependencies by running

   ```bash
   npm i
   ```

3. **Run the Development Server**

   You can now run the development server for the app with

   ```bash
   npm run dev
   ```

   and visit it on [http://localhost:5173](http://localhost:5173)
