# Sign Language Recognition using Triple-Stream I3D

This repository contains the implementation of a Sign Language Recognition system using Triple-Stream Inflated 3D ConvNet (I3D) architecture. The model is trained and evaluated on the UCF101 dataset, demonstrating its effectiveness in action recognition tasks.

## Features

- Implementation of Triple-Stream I3D architecture
- Support for multiple input modes (RGB, Grayscale, Optical Flow)
- Flexible input frame size and sequence length
- Built with TensorFlow/Keras
- Training visualization with loss and accuracy plots
- Model evaluation and saving capabilities

## Prerequisites

```bash
- Python 3.x
- TensorFlow
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the UCF101 dataset, which contains:
- 101 action categories
- 13,320 videos
- Videos collected from YouTube
- Challenging variations in camera motion, object appearance, pose, scale, and lighting conditions

## Usage

### Training

You can train the model using the following command:

```bash
python train.py --batch 64 --epoch 10 --shape_image 128 --num_frame 10 --nclass 24 --mode 2
```

### Parameters

- `--batch`: Batch size for training (default: 64)
- `--epoch`: Number of training epochs (default: 10)
- `--shape_image`: Input frame size (default: 128)
- `--num_frame`: Number of frames per sequence (default: 10)
- `--nclass`: Number of output classes (default: 24)
- `--freeze`: Freeze layers option (0: no, 1: yes)
- `--mode`: Input mode (3: RGB, 1: Grayscale, 2: Optical Flow)
- `--data_format`: Data format (1: channels_last, 0: channels_first)

## Model Architecture

The implementation is based on the I3D (Inflated 3D ConvNet) architecture, which is designed to effectively learn spatiotemporal features from video data. The model includes:

- 3D Convolutional layers
- Batch Normalization
- MaxPooling3D layers
- Dropout for regularization
- Dense layers for classification

## Training Visualization

The training process includes visualization of:
- Training and validation accuracy
- Training and validation loss

## Model Evaluation

The model's performance is evaluated using:
- Test accuracy
- Loss metrics
- Confusion matrix (optional)

## Saving the Model

After training, the model is automatically saved as 'I3D_OPT.h5' in the project directory.

## Reference

This implementation is based on the paper "Sign Language Recognition base Triple-Steam I3D" and incorporates concepts from:
- ["Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"](https://arxiv.org/abs/1705.07750)

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contributing

Feel free to submit issues and enhancement requests!
