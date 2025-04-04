Human Emotion Prediction Model (WIP)

This is a work-in-progress project where I’m building a human emotion prediction model using PyTorch and TensorFlow. The idea is to feed in facial data (probably images or frames from videos) and have the model classify the emotion being expressed — like happy, sad, angry, surprised, etc.
🔧 Stack / Tools Being Used

PyTorch — for building and training custom CNN architectures
TensorFlow/Keras — mainly for experimentation, visualization, and maybe even model comparison
OpenCV — for face detection and image preprocessing
NumPy + Pandas — the usual data wrangling stuff
Matplotlib/Seaborn — visualization of metrics/losses
Scikit-learn — might use this for accuracy, confusion matrix, etc.
🧠 Model Architecture (WIP)

Currently prototyping with layered CNNs. The idea is to start simple and keep stacking.
Layers I'm Playing With:
Conv2D layers with ReLU activations
MaxPooling to reduce spatial dimensions
BatchNorm for more stable training
Dropout to prevent overfitting
Flatten + Dense for the final classification
Softmax at the end for multi-class prediction
Trying different combinations and depths. May integrate a ResNet or MobileNet later down the line depending on accuracy/performance trade-offs.
🧪 What It Does (Eventually)

The goal is:
Detect human faces from images or webcam
Preprocess and normalize the input
Feed into model
Output emotion label (like “happy”, “sad”, “angry”, etc.)
📦 Dataset

Still exploring datasets. Right now experimenting with:
FER2013
Will possibly augment using custom data or scrape from open-source resources
Data is being normalized, resized to 48x48 or 96x96, and split into train/val/test sets.
🧪 Current Status

 Basic CNN model working in PyTorch
 Data pipeline mostly set up
 Training needs tuning (accuracy not great yet)
 TensorFlow version in progress
 Need to integrate real-time webcam prediction
 Might deploy to a web app down the line


 🚧 What’s Next

Clean up training loops
Optimize validation accuracy
Add confusion matrix + metrics reporting
Hook up webcam input with real-time emotion inference
Deploy version (maybe Flask/Streamlit)
