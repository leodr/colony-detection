# Annotation app

Web app for labeling colonies with instance segmentation masks. Clicking a colony masks it with the Segment Anything Model (SAM). See [explanation.mp4](explanation.mp4) for a walkthrough.

The app has two parts: a Python server (`main.py`) that runs SAM, and a web interface (`interface/`).

## Server

1. **Install the requirements** (Python 3)

   ```bash
   pip install -r requirements.txt
   ```

2. **Install PyTorch**

   Follow the instructions [here](https://pytorch.org/get-started/locally/). If you have an NVIDIA GPU, choose the CUDA version.

3. **Download the SAM model file**

   Download the ViT-H checkpoint into this folder as `sam_vit_h.pth`. If the link is dead, check the [Segment Anything repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

   ```bash
   curl -o sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

4. **Run the server** from this folder

   ```bash
   python3 main.py
   ```

## Interface

1. **Install [Node.js](https://nodejs.org/en)** (LTS release)

2. **Install the dependencies** in `interface/`

   ```bash
   npm i
   ```

3. **Run the development server**

   ```bash
   npm run dev
   ```

   and open [http://localhost:5173](http://localhost:5173).

## Docker

`docker-compose.yml` builds both parts into one image and serves the app on port 80. Labeled masks are written to the `annotation_app_output` volume.
