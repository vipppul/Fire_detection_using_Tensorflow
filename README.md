# Fire_detection_using_Tensorflow

# 🔥 Fire Detection(IMG) using Deep Learning (MobileNetV2)

This project implements a fire and no-fire image classification system using TensorFlow and Keras with a MobileNetV2-based model. It supports cross-validation, metrics tracking, and final evaluation on held-out data.

## 🚀 Features

- Binary classification: **fire** 🔥 vs **no fire** 🧯
- Model architecture: MobileNetV2 (Transfer Learning)
- Cross-validation with performance metrics
- Training visualization with plots
- Final model evaluation with classification report
- Custom F1-score callback tracking
- Inference on custom images

---

## 📁 Project Structure

- Fire_models      -> contains the model and the graphs.
- eval_mod.py      -> is the python script used for model training and plotting the graphs you can use this script just change the paths if needed.
- f1_callback.py   -> contain code for the callback.
- run_fire_mod.py  -> contains code to run the keras model.
- requirements.txt -> required libraries listed that are need to be pip installed. (pip install -r requirements.txt)

## 🧠 Model Info

- Architecture: MobileNetV2 (transfer learning)
- Input Shape: (128, 128, 3)
- Final Activation: Sigmoid (binary classification)
- Loss: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall, AUC, F1 Score