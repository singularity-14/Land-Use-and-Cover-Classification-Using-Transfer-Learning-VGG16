# 🛰️ Land Use and Land Cover Classification using Transfer Learning with VGG16

> A transfer learning pipeline leveraging pre-trained **VGG16** for multi-class **Land Use and Land Cover (LULC) classification** on the EuroSAT satellite imagery dataset—achieving accurate classification of 10 diverse land cover types using frozen convolutional layers and custom classification heads.

---

## 📌 Project Overview

This project implements a **Transfer Learning approach** for satellite image classification using the VGG16 convolutional neural network. The pre-trained ImageNet weights are leveraged to extract high-level spatial features from satellite imagery, then fine-tuned with custom classification layers to identify 10 distinct land use and land cover classes. The model is trained on the **EuroSAT dataset**, a collection of Sentinel-2 satellite images with diverse geographical and seasonal variations.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 Transfer Learning | Pre-trained VGG16 (ImageNet weights) as feature extractor |
| 🔒 Frozen Layers | Convolutional base layers frozen to retain ImageNet knowledge |
| 🎯 Multi-class Classification | 10 LULC classes from satellite imagery |
| 📊 Data Augmentation | ImageDataGenerator with rotation, shifting, zooming, and shearing |
| 📈 Training Monitoring | Validation split (80/20) with accuracy and loss tracking |
| 📉 Performance Visualization | Line plots for training/validation accuracy and loss curves |
| 💾 Model Persistence | Trained model saved in Keras format |

---

## 🚀 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, Scikit-image |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset | EuroSAT Sentinel-2 Satellite Images |

---

## 📂 Dataset

**Source:** [EuroSAT Dataset](https://github.com/phelber/eurosat)

The EuroSAT dataset consists of high-resolution satellite images from the **Copernicus Sentinel-2 satellite** mission.

| Aspect | Details |
|--------|---------|
| **Image Format** | RGB (64×64 pixels) |
| **Spectral Bands** | 13 bands (reduced to RGB for VGG16) |
| **Classes** | 10 LULC categories |
| **Data Split** | 80% Training / 20% Validation |
| **Batch Size** | 32 images per batch |

### Land Use and Land Cover Classes

| Class | Category |
|-------|----------|
| 1 | Urban |
| 2 | Agricultural |
| 3 | Forest |
| 4 | Herbaceous |
| 5 | Industrial |
| 6 | Permanent Crop |
| 7 | Pasture |
| 8 | Bare Rock |
| 9 | Water |
| 10 | Glaciers / Snow |

---

## 🏗️ Architecture

### VGG16 Base Model
- **Frozen Feature Extractor:** 13 convolutional layers from pre-trained ImageNet weights
- **Input Shape:** (64, 64, 3) - RGB satellite images
- **Pre-training:** ImageNet

### Custom Classification Head
```
VGG16 Base (Frozen)
    ↓
Flatten Layer
    ↓
Dense (256 neurons, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Output Dense (10 neurons, Softmax activation)
```

---

## 📊 Data Preprocessing & Augmentation

**ImageDataGenerator Configuration:**

| Transformation | Parameter |
|---|---|
| Rescaling | 1/255 (normalization) |
| Width Shift | 20% |
| Height Shift | 20% |
| Shear Range | 0.2 |
| Zoom Range | 0.2 |
| Fill Mode | Nearest |

**Purpose:** Increase dataset diversity and reduce overfitting on the satellite imagery dataset.

---

## 🏋️ Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Metric** | Accuracy |
| **Epochs** | 20 |
| **Batch Size** | 32 |
| **Validation Strategy** | 20% split |
| **Frozen Layers** | All convolutional layers in VGG16 |

---

## 📈 Model Performance

### Training Results

The model is trained for **20 epochs** on the EuroSAT dataset with the configuration above.

**Performance Tracking:**
- Training accuracy increases progressively across epochs
- Validation accuracy monitored to detect overfitting
- Training loss and validation loss plotted for convergence analysis

### Output Visualization

1. **Accuracy Curve:** Training vs. Validation accuracy over 20 epochs
2. **Loss Curve:** Training vs. Validation loss over 20 epochs

Both curves demonstrate model convergence and generalization capability.

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/singularity-14/Land-Use-and-Cover-Classification-Using-Transfer-Learning-VGG16.git
cd Land-Use-and-Cover-Classification-Using-Transfer-Learning-VGG16
```

### 2. Install dependencies
```bash
pip install tensorflow opencv-python pandas numpy matplotlib seaborn scikit-image
```

### 3. Prepare the dataset
```bash
# Download EuroSAT dataset from: https://github.com/phelber/eurosat
# Organize structure:
# data/
#   ├── class_1/
#   ├── class_2/
#   ├── ...
#   └── class_10/
```

### 4. Update file paths in the notebook
```python
directory = r'/path/to/your/data'  # Update with your dataset path
```

### 5. Run the notebook
```bash
jupyter notebook lulc.ipynb
```

### 6. Use the trained model for predictions
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('multiclass_land_use_land_cover.keras')

# Prepare an image
img = image.load_img('satellite_image.png', target_size=(64, 64))
img_array = image.img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_batch)
predicted_class = np.argmax(prediction, axis=1)
confidence = np.max(prediction)

print(f"Predicted Class: {predicted_class[0]}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📂 Project Structure

```
Land-Use-and-Cover-Classification-Using-Transfer-Learning-VGG16/
│
├── lulc.ipynb                              # Main notebook with full pipeline
├── multiclass_land_use_land_cover.keras   # Trained model (saved after training)
├── README.md                               # Project documentation
└── LICENSE                                 # MIT License
```

---

## 💡 Key Learnings & Takeaways

- **Transfer Learning Power:** Pre-trained ImageNet features are highly effective for satellite image classification without retraining entire CNNs
- **Frozen Convolutional Layers:** Retaining ImageNet weights prevents overfitting on smaller satellite datasets while leveraging learned features
- **Data Augmentation Necessity:** Augmentation techniques (rotation, shifting, zoom) significantly improve model robustness and generalization
- **Batch Normalization Impact:** The custom Dense layers with Dropout(0.5) help regularize and prevent overfitting
- **20 Epochs Sufficiency:** Training converges effectively within 20 epochs on augmented EuroSAT data
- **Multiclass Softmax:** Categorical crossentropy + softmax activation appropriate for 10-class satellite image classification
- **Validation Monitoring:** 80/20 train-validation split effectively detects overfitting and ensures reliable performance metrics

---

## 🔗 Dataset Reference

**EuroSAT:** Land Use Classification with Sentinel-2  
Repository: https://github.com/phelber/eurosat  
Paper: Helber, P., et al. (2019). "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*A transfer learning approach to satellite image classification, demonstrating the effectiveness of pre-trained deep neural networks for land use and land cover analysis with EuroSAT Sentinel-2 imagery.*

3. **Evaluation**:
   - Evaluate the model performance based on training and validation accuracy and loss plots.

## Customization

- Experiment with different pre-trained models and architectures.
- Fine-tune hyperparameters such as learning rate, dropout rate, and batch size.
- Explore advanced data augmentation techniques for improving model robustness.

## File Structure

- `lulc.ipynb`: Main Python script for model training and evaluation.
- `README.md`: This file providing an overview and instructions.
- `data/`: Directory containing the EuroSAT dataset split into training and validation sets.

## License

This project is licensed under the [MIT License](LICENSE).