import streamlit as st
import numpy as np
from tensorflow import keras
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os
import time
from glob import glob
from collections import defaultdict

# Initialize challenge stats
if "fool_successes" not in st.session_state:
    st.session_state.fool_successes = 0
if "model_successes" not in st.session_state:
    st.session_state.model_successes = 0
if "challenge_history" not in st.session_state:
    st.session_state.challenge_history = []

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="sureshkrishnan/huggingface_mnist_upload",
        filename="mnist_model.keras"
    )
    return keras.models.load_model(model_path)

model = load_model()

# Sidebar: brush settings and identity
st.sidebar.title("🖌️ Brush Settings")
stroke_width = st.sidebar.slider("Stroke width", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Stroke color", "#FFFFFF")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by **Suresh**")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Draw", "🖼️ Gallery", "🧠 Model Info", "🧪 Challenge Mode"])

# Preprocessing function
def preprocess_image(image_data):
    img = Image.fromarray(image_data.astype('uint8'))
    img = ImageOps.grayscale(img).resize((28, 28))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 28, 28, 1)

# DRAW TAB
with tab1:
    st.title("🎨 Draw a Digit")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_draw"
    )

    if canvas_result.image_data is not None:
        img_array = preprocess_image(canvas_result.image_data)

        if np.sum(img_array) < 0.1:
            st.warning("🖼️ Please draw a digit before predicting.")
        else:
            with st.spinner("🔍 Predicting..."):
                prediction = model.predict(img_array)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit]

            st.success(f"### 🔢 Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f})")
            st.bar_chart(prediction[0])

            # Top-3 predictions
            st.markdown("### 🔝 Top Predictions:")
            top_indices = prediction[0].argsort()[-3:][::-1]
            for i in top_indices:
                st.write(f"🔢 {i}: {prediction[0][i]:.2f}")

            # Label and save to custom dataset
            st.markdown("### 🧑‍🏫 Label Your Digit")
            label = st.selectbox("Choose the correct label:", list(range(10)))
            if st.button("📥 Save to Custom Dataset"):
                os.makedirs("custom_digits", exist_ok=True)
                img = Image.fromarray((img_array[0].reshape(28, 28) * 255).astype(np.uint8))
                img.save(f"custom_digits/{label}_{int(time.time())}.png")
                st.success(f"Saved your drawing as digit '{label}'!")

            # Send to Challenge Mode
            if st.button("🚀 Send to Challenge Mode"):
                st.session_state.challenge_image = canvas_result.image_data
                st.success("Drawing sent to Challenge Mode! Switch to the Challenge tab.")

    if st.button("🧹 Clear Canvas", key="clear_draw"):
        st.experimental_rerun()

# GALLERY TAB
with tab2:
    st.title("🖼️ Custom Digit Gallery")

    image_files = sorted(glob("custom_digits/*.png"), reverse=True)

    if image_files:
        grouped = defaultdict(list)
        for file in image_files:
            filename = os.path.basename(file)
            label = filename.split("_")[0]
            grouped[label].append(file)

        for label in sorted(grouped.keys(), key=int):
            st.markdown(f"### 🔢 Label: {label} ({len(grouped[label])} images)")
            cols = st.columns(5)
            for i, file in enumerate(grouped[label][:5]):
                with cols[i % 5]:
                    st.image(file, width=100, caption=os.path.basename(file))
    else:
        st.info("No custom digits saved yet.")

# MODEL INFO TAB
with tab3:
    st.title("🧠 How the Model Works")
    st.markdown("""
    - Trained on the MNIST dataset (28×28 grayscale digits)
    - Uses a simple CNN architecture
    - Hosted on Hugging Face for easy access
    - Predictions are based on pixel intensity and shape
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="MNIST Samples")

# CHALLENGE MODE TAB
with tab4:
    st.title("🧪 Challenge the Model")
    st.markdown("Try to fool the model! Use a drawing from the Draw tab and label what you intended.")

    if "challenge_image" in st.session_state:
        img_array = preprocess_image(st.session_state.challenge_image)

        if np.sum(img_array) < 0.1:
            st.warning("🖼️ Drawing is too faint. Try again.")
        else:
            with st.spinner("🔍 Predicting..."):
                prediction = model.predict(img_array)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit]

            st.write(f"### 🤖 Model Prediction: {predicted_digit} (Confidence: {confidence:.2f})")

            challenge_label = st.selectbox("What digit did you intend to draw?", list(range(10)))
            if st.button("✅ Submit Challenge"):
                if challenge_label != predicted_digit:
                    st.session_state.fool_successes += 1
                    result = "Fooled"
                    st.success("😈 You fooled the model!")
                else:
                    st.session_state.model_successes += 1
                    result = "Correct"
                    st.info("🤖 Model got it right!")

                st.session_state.challenge_history.append({
                    "label": challenge_label,
                    "prediction": predicted_digit,
                    "result": result,
                    "confidence": float(confidence)
                })

            # Score display
            st.markdown("### 🧮 Challenge Score")
            st.write(f"😈 Fooled the Model: {st.session_state.fool_successes}")
            st.write(f"🤖 Model Got It Right: {st.session_state.model_successes}")

            # Leaderboard
            st.markdown("### 🏆 Leaderboard (Top Wins by Confidence)")
            wins = [h for h in st.session_state.challenge_history if h["result"] == "Fooled"]
            if wins:
                top_wins = sorted(wins, key=lambda x: -x["confidence"])[:5]
                for i, win in enumerate(top_wins, 1):
                    st.write(f"{i}. Intended: {win['label']} | Predicted: {win['prediction']} | Confidence: {win['confidence']:.2f}")
            else:
                st.info("No wins yet. Try to fool the model!")

            # Trend chart
            st.markdown("### 📈 Win/Loss Trend")
            trend_data = {
                "Fooled": st.session_state.fool_successes,
                "Correct": st.session_state.model_successes
            }
            st.bar_chart(trend_data)

            # Reset button
            if st.button("🔄 Reset Challenge Stats"):
                st.session_state.fool_successes = 0
                st.session_state.model_successes = 0
                st.session_state.challenge_history = []
                st.success("Challenge stats reset!")
    else:
        st.info("No drawing received yet. Go to the Draw tab and click 'Send to Challenge Mode'.")
