## FaceGeo
This project develops a model to reconstruct an individual's facial geometry using six portrait images captured from different viewpoints.
### Input
- Required: Six multi-view images of the face.
- Optional: The model can still function with fewer than six images, though accuracy may decrease.
- For Best Results: Include images from the following angles: front, left, right, up, and bottom.
  
💁  Note: You do not need to provide camera poses, nor is it necessary to use fixed camera positions.

### Output
- The output is a face mesh, which is saved in OBJ file format.

### Network Architecture
![network_architecture](data/network.png)

The model employs an encoder-fuser-decoder architecture, consisting of the following components:
- **Encoder**: ResNet18
- **Fuser**: Transformer Decoder
- **Decoder**: UNet

🔔 **Colab Tutorial for inference**: click [here](https://colab.research.google.com/github/yankuai/FaceGeo/blob/main/FaceGeo-reconstruct-face-geometry-with-cross-attention.ipynb).

### Environment
```sh
git clone https://github.com/yankuai/FaceGeo.git
cd FaceGeo
pip install -r requirements.txt
```
### Inference
You can perform inference on a folder containing multiple individuals' face images. The data should be organized in the following structure:

```sh
input/
├── subject_1
│   ├── view_1.png
│   ├── ...
│   └── view_k.png
├── ...
└── subject_n
      ├── view_1.png
      ├── ...
      └── view_k.png
```

**Steps to Run Inference:**

1. Download Pre-trained Weights: Download the pre-trained [weights](https://github.com/yankuai/FaceGeo/releases/download/v1.0.0/model.pt) into the base folder and the file to `model.pt`.

2. Modify Configuration File: Update the following fields in the configuration file `conf/inference.yaml`:
- `home_dir`
- `out_dir`
- `data_dir`
- `checkpoint`

3. Run Inference Command:

```sh
python inference.py conf/inference.yaml
```
