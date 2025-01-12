# Wheat Disease Classification Project

This repository contains a convolutional neural network (CNN)-based approach for classifying different types of wheat diseases (and healthy wheat) using transfer learning with **VGG19**. The project was developed in Google Colab and uses a dataset stored in Google Drive. 

The repository includes:
1. Data loading and preprocessing from Google Drive
2. Building a CNN architecture on top of a pre-trained **VGG19**
3. Training the network and evaluating performance
4. Visualizing training accuracy/loss and confusion matrix
5. Instructions for reproducing the work on your own dataset

---

## Table of Contents

- [Overview](#overview)  
- [Dataset Description](#dataset-description)  
- [Prerequisites and Installation](#prerequisites-and-installation)  
- [Project Structure](#project-structure)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Training and Evaluation](#training-and-evaluation)  
  - [Training Curves](#training-curves)  
  - [Confusion Matrix](#confusion-matrix)  
- [Results](#results)  
- [Customization](#customization)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This project aims to classify four types of wheat conditions:

1. **Crown and Root Rot**  
2. **Healthy Wheat**  
3. **Leaf Rust**  
4. **Wheat Loose Smut**

By leveraging transfer learning with **VGG19**, we can effectively use a pre-trained model on ImageNet and fine-tune the network for our specific classification task. The code is written in Python (compatible with Google Colab) and uses frameworks/libraries such as:

- **TensorFlow/Keras**  
- **OpenCV**  
- **Scikit-Learn**  
- **Matplotlib** & **Seaborn**  
- **NumPy**  

---

## Dataset Description

The dataset is stored in Google Drive under the path:
```
/content/drive/MyDrive/Large Wheat classification detection/
```
Within this folder, there are subdirectories for each class:
- `Crown and Root Rot/`
- `Healthy Wheat/`
- `Leaf Rust/`
- `Wheat Loose Smut/`

Each directory contains images of the corresponding wheat disease (or healthy samples).

If you want to use your own dataset:
1. Create a similar directory structure in your Google Drive (or local).
2. Update the paths in the code accordingly.

---

## Prerequisites and Installation

1. **Google Account**: Required for Google Colab and Google Drive.  
2. **Python 3.7+**: If running locally, ensure you have Python 3.7 or above.  
3. **Libraries**:  
   - `numpy`  
   - `matplotlib`  
   - `seaborn`  
   - `opencv-python` (OpenCV)  
   - `scikit-learn`  
   - `tensorflow` and/or `keras`  
   - `tqdm`  
   - `imutils`  

In Google Colab, most libraries are pre-installed. You may need to install missing libraries via `!pip install <library-name>`.

---

## Project Structure

Below is a suggested file structure if you clone or download this repository. The structure may vary depending on your personal setup.

```
wheat-disease-classification/
├── README.md
├── wheat_classification.ipynb     # The main notebook (Google Colab or Jupyter)
├── lb.pickle                     # Label binarizer (after training)
└── data/
    ├── Crown and Root Rot/
    ├── Healthy Wheat/
    ├── Leaf Rust/
    └── Wheat Loose Smut/
```

- **wheat_classification.ipynb**: Contains all code for data loading, model building, training, and evaluation.
- **lb.pickle**: Saved `LabelBinarizer` object for encoding/decoding class labels (created after training).
- **saved_model.h5**: (Optional) A checkpoint or final trained model (HDF5 format).  

---

## Usage

1. **Clone or Download the Repository**  
   ```bash
   git clone https://github.com/your-username/wheat-disease-classification.git
   cd wheat-disease-classification
   ```

2. **Open in Google Colab (Recommended)**  
   - Upload the repository or the `.ipynb` file to your Google Drive or GitHub.
   - Open the `.ipynb` in Google Colab.
   - Make sure to mount your Google Drive to access the dataset.

3. **Set the Dataset Paths**  
   - Update the paths for your own dataset if necessary. Example paths:
     ```python
     dataset = "/content/drive/MyDrive/Large Wheat classification detection"
     CROWN_AND_ROOT_ROT_PATH = "/content/drive/MyDrive/Large Wheat classification detection/Crown and Root Rot"
     HEALTHY_AND_WHEAT_PATH = "/content/drive/MyDrive/Large Wheat classification detection/Healthy Wheat"
     LEAF_RUST_PATH = "/content/drive/MyDrive/Large Wheat classification detection/Leaf Rust"
     WHEAT_LOOSE_SMUT_PATH = "/content/drive/MyDrive/Large Wheat classification detection/Wheat Loose Smut"
     ```

4. **Install Dependencies (if needed)**  
   ```python
   !pip install opencv-python imutils tqdm seaborn
   ```

5. **Run All Cells**  
   - The notebook will:
     1. Mount Google Drive (if in Colab).  
     2. Load and preprocess the images.  
     3. Create train/test splits.  
     4. Perform data augmentation.  
     5. Build the CNN using VGG19 as a backbone.  
     6. Train the model.  
     7. Evaluate and display the confusion matrix.  

---

## Model Architecture

We use **VGG19** pre-trained on ImageNet as our base, then we add custom layers:
1. **AveragePooling2D**  
2. **Flatten**  
3. **Dense (512 neurons, ReLU activation)**  
4. **Dropout (0.2)**  
5. **Dense** layer with `softmax` for multi-class classification.

```python
headmodel = VGG19(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

model = headmodel.output
model = AveragePooling2D(pool_size=(5, 5))(model)
model = Flatten(name='flatten')(model)
model = Dense(512, activation='relu')(model)
model = Dropout(0.2)(model)
model = Dense(len(Labels), activation='softmax')(model)

final_model = Model(inputs=headmodel.input, outputs=model)
```

All layers in the base VGG19 model are frozen during the initial training phase.

---

## Training and Evaluation

### Training Curves

We train the top layers for 30 epochs with the following configurations:
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam`
- **Learning Rate**: default 0.001
- **Metrics**: `accuracy`

**Data Augmentation** with `ImageDataGenerator` includes:
- Rotation up to 30 degrees
- Zoom up to 0.15
- Width/height shift range
- Shear range
- Horizontal flip
- Mean subtraction

After training, we plot the **Training vs. Validation Accuracy** and **Loss**:

```python
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
...
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
...
```

### Confusion Matrix

We compute the confusion matrix to see class-level performance:

```python
true_labels = np.argmax(testY, axis=1)
predicted_labels = np.argmax(final_model.predict(testX), axis=1)

conf_matrix = confusion_matrix(true_labels, predicted_labels)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=Labels, yticklabels=Labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

---

## Results

- The model typically achieves high accuracy (the exact number depends on your dataset size and quality).  
- The confusion matrix helps visualize any classes the model struggles to differentiate.  

Feel free to tune hyperparameters (learning rate, epochs, batch size) or unfreeze deeper layers of VGG19 for better performance at the cost of increased training time.

---

## Customization

1. **New Classes**  
   - Add or remove disease categories by creating new directories and placing images there.  
   - Update `LABELS` to match your class names.  
2. **Data Augmentation**  
   - Modify rotation/zoom/shift/flip parameters in the `ImageDataGenerator`.  
3. **Training Parameters**  
   - Change **batch size**, **epochs**, or **optimizer**.  
   - Unfreeze more layers in VGG19 for fine-tuning.  
4. **Inference/Prediction**  
   - Once the model is trained, you can do `model.predict()` on new images to classify them.

---

## Contributing

Contributions are welcome! If you find a bug or want to propose a new feature:
1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/my-new-feature`).  
3. Commit your changes (`git commit -m 'Add some feature'`).  
4. Push to the branch (`git push origin feature/my-new-feature`).  
5. Create a Pull Request.  

---

## License

This project is open-sourced under the [MIT License](LICENSE). You are free to use, modify, and distribute this project as you see fit. 

Please cite the source if you use it in your own work. 

---

**Thank you for checking out this Wheat Disease Classification project!**  
For any questions or clarifications, feel free to open an [issue](../../issues) or reach out. Contributions, suggestions, and improvements are greatly appreciated.
