```markdown
# Dogs vs. Cats Image Classification

This repository contains code for a simple image classification project that distinguishes between images of dogs and cats. It uses a convolutional neural network (CNN) to perform the classification task. You can train and evaluate the model with different configurations using the provided script.

## Setup

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/TimChou-ntu/dogs_cats.git
   cd dogs_cats
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset. You should have a directory containing subdirectories for the "train" dataset, with images of dogs and cats.

   Example directory structure:

   ```
   ./dogs-vs-cats
   ├── data
   │   └── train
   │       └── train
   │           ├── dog.1.jpg
   │           ├── dog.2.jpg
   │           ├── cat.1.jpg
   │           ├── cat.2.jpg
   │           └── ...
   ```

## Configuration

You can configure various aspects of the training process by using command-line arguments. Here's a list of available configuration options and their descriptions:

<!-- - `--config`: Path to the configuration file. -->
- `--exp-name`: Experiment name.
- `--accelerator`: Accelerator to use (`auto`, `gpu`, or `cpu`).
- `--seed`: Seed for initializing training.
- `--dataset-path`: Path to the dataset.
- `--batch-size`: Input batch size for training.
- `--learning-rate`: Learning rate.
- `--gamma`: Learning rate step gamma.
- `--max-epochs`: Number of epochs to train.
- `--image_output_path`: Path to save images.
- `--eval`: Evaluate the model on the validation set.
- `--ckpt-path`: Path to the checkpoint folder or a direct path to a checkpoint file.

## Training

To train the model, use the following command:

```bash
python train.py --dataset-path /path/to/your/dataset/
```

The way I use is 

```bash
python train.py --dataset-path ./dogs-vs-cats/train/train/
```

You can customize the training by passing the desired configuration options as arguments. For example, to change the batch size and learning rate:

```bash
python train.py --dataset-path /path/to/your/dataset/ --batch-size 128 --learning-rate 0.001
```

## Validation

To validate the trained model, you can use the following command:

```bash
python train.py --dataset-path /path/to/your/dataset/ --eval
```

You can also specify a specific checkpoint file or folder using the `--ckpt-path` option.

## Performance of Default Configuration
The performance of the default configuration is as follows:

- Validation Precision: 0.9425
- Validation Accuracy: 0.9500
- Validation Recall: 0.9556
- Validation Loss: 0.1238 (cross entropy loss)

## Reproducibility

To ensure reproducibility, the script sets a seed (42) using the `seed_everything` function.

```python
seed_everything(args.seed)
```

## Acknowledgments

- This project uses the PyTorch Lightning library for deep learning.
- Dataset: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

Feel free to explore the code and experiment with different configurations to improve the classification model. Good luck with your image classification project!
