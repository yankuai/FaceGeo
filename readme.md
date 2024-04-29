## FaceGeo
👶 This project builds a model to reconstruct an indivisual's face geometry given 6 portrait images captured from diffirent views.
- Input: 6 multi-view images of the face. 

    💁 If you have less than 6 images, the model can also work, but the accuracy can be low.

    💁 The input images should include {front, left, right, up, bottom} views of the face for the best accuracy. You don't need to provide their camera poses or always use fixed camera poses.
- Output: A face mesh saved as an OBJ file.


🚀 The network has an encoder-fuser-decoder architecture.
- Encoder: Resnet18
- Fuser: Transformer decoder
- Decoder: UNet

🔔 We only release the code for inference. Colab Tutorial: click (here)[].

### Environment
```sh
pip install -r requirements.txt
```
### Inference
Inference on a folder of multiple indivisuals' face images.

- The data should be arranged into the following structure.

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

- Download the pre-trained (weights)[] into base folder and rename it as `model.pt`.

- Then modify the config file `conf/inference.yaml`.

- Run the command:

```sh
python inference.py
```
