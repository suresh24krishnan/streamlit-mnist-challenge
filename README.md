# streamlit-mnist-challenge
Interactive Streamlit app for digit recognition using a CNN trained on MNIST. Draw, predict, and challenge the model.
# 🧪 Streamlit MNIST Challenge App

An interactive Streamlit app where users draw digits, test a CNN model trained on MNIST, and try to fool it! Built with TensorFlow, Hugging Face, and Streamlit — perfect for demos, education, and playful experimentation.

## 🚀 Features

- 🎨 **Freehand Digit Drawing**: Use a canvas to sketch digits in real time.
- 🧠 **Model Predictions**: View top predictions with confidence scores and bar charts.
- 🖼️ **Custom Digit Gallery**: Save and browse your labeled drawings.
- 🧪 **Challenge Mode**: Try to fool the model and track your wins on a leaderboard.
- 📈 **Trend Charts**: Visualize your success rate against the model.

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) for UI
- [TensorFlow](https://www.tensorflow.org/) for digit classification
- [Hugging Face Hub](https://huggingface.co/) for model hosting
- [Pillow](https://python-pillow.org/) for image processing
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) for drawing interface

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/streamlit-mnist-challenge.git
cd streamlit-mnist-challenge
pip install -r requirements.txt

## 🙌 Acknowledgments

- MNIST dataset by Yann LeCun et al.
- Hugging Face for model hosting
- Streamlit community for awesome widgets
