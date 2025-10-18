import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

st.set_page_config(page_title="Image Lab", layout="wide")

st.title("🎨 Image Processing Lab")
st.write("Upload an image and apply various filters and transformations.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

if uploaded_file:
    # Original Image
    original_img = Image.open(uploaded_file).convert("RGB")
    edited_img = original_img.copy()

    # Tabs for different editing tools
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Adjustments", "🧪 Filters", "🎨 Color Effects", "⬇️ Download"])

    # ---- TAB 1: BASIC ADJUSTMENTS ----
    with tab1:
        st.header("🔧 Basic Adjustments")

        brightness = st.slider("Brightness", 0.5, 2.5, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.5, 5.0, 1.0, 0.1)

        enhancer = ImageEnhance.Brightness(edited_img)
        edited_img = enhancer.enhance(brightness)

        enhancer = ImageEnhance.Sharpness(edited_img)
        edited_img = enhancer.enhance(sharpness)

    # ---- TAB 2: FILTERS ----
    with tab2:
        st.header("🧪 Apply Filters")

        filter_type = st.selectbox("Choose a filter", ["None", "Blur", "Edge Detection", "Emboss", "Contour"])

        img_cv = pil_to_cv2(edited_img)

        if filter_type == "Blur":
            k = st.slider("Blur intensity", 1, 25, 5, step=2)
            img_cv = cv2.GaussianBlur(img_cv, (k, k), 0)
        elif filter_type == "Edge Detection":
            img_cv = cv2.Canny(img_cv, 100, 200)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif filter_type == "Emboss":
            kernel = np.array([[ -2, -1, 0],
                               [ -1,  1, 1],
                               [  0,  1, 2]])
            img_cv = cv2.filter2D(img_cv, -1, kernel)
        elif filter_type == "Contour":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30, 100)
            img_cv = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        edited_img = cv2_to_pil(img_cv)

    # ---- TAB 3: COLOR EFFECTS ----
    with tab3:
        st.header("🎨 Color Effects")

        effect = st.radio("Choose color effect", ["None", "Grayscale", "Negative", "Sepia"], horizontal=True)
        img_cv = pil_to_cv2(edited_img)

        if effect == "Grayscale":
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif effect == "Negative":
            img_cv = 255 - img_cv
        elif effect == "Sepia":
            img_cv = apply_sepia(img_cv)

        edited_img = cv2_to_pil(img_cv)

    # ---- IMAGE PREVIEW SECTION ----
    st.markdown("### 🖼️ Preview: Original vs Edited")

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="📸 Original Image", use_column_width=True)
    with col2:
        st.image(edited_img, caption="🧪 Edited Image", use_column_width=True)

    # ---- TAB 4: DOWNLOAD ----
    with tab4:
        st.header("⬇️ Download Your Edited Image")
        buf = io.BytesIO()
        edited_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button("📥 Download", data=byte_im, file_name="edited_image.png", mime="image/png")
else:
    st.info("👆 Upload an image to get started.")
