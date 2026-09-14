import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

st.set_page_config(
    page_title="Image Classification System",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    background: #164A70;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
.result-box {
    background: #F4F6F7;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #D5D8DC;
}
.prediction-card {
    background: #E8EEF3;
    color: #1F2937;
    padding: 15px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 18px;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #666;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>IMAGE CLASSIFICATION</h1>
    <p>AI-Based Image Classification System</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet", include_top=True)


def get_proper_object_name(class_name):
    """Convert ImageNet class labels into a clean display name."""
    name = class_name.replace("_", " ").strip()

    # Common ImageNet names that should be displayed cleanly.
    special_names = {
        "golden retriever": "Golden Retriever",
        "labrador retriever": "Labrador Retriever",
        "german shepherd": "German Shepherd",
        "english springer": "English Springer",
        "cocker spaniel": "Cocker Spaniel",
        "african elephant": "African Elephant",
        "indian elephant": "Indian Elephant",
        "tiger shark": "Tiger Shark",
        "great white shark": "Great White Shark",
        "siamese cat": "Siamese Cat",
        "persian cat": "Persian Cat",
        "tabby cat": "Tabby Cat",
        "egyptian cat": "Egyptian Cat",
    }

    return special_names.get(name.lower(), name.title())


def classify_image(image):
    image = image.resize((224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    model = load_model()
    prediction = model.predict(image_array, verbose=0)

    return decode_predictions(prediction, top=5)[0]


with st.sidebar:
    st.header("About Project")
    st.write(
        "This BCA Minor Project uses the pre-trained MobileNetV2 "
        "deep learning model to classify images."
    )

    st.markdown("### Technology")
    st.write("• Python")
    st.write("• Streamlit")
    st.write("• TensorFlow")
    st.write("• MobileNetV2")
    st.write("• ImageNet")


st.subheader("Choose an Image")

uploaded_file = st.file_uploader(
    "Select any JPG, JPEG, PNG, BMP or WEBP image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is None:
    st.info(
        "Waiting for an image... Upload an image to start "
        "the AI classification."
    )
else:
    try:
        selected_image = Image.open(uploaded_file).convert("RGB")
        width, height = selected_image.size

        st.subheader("Image Preview")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                selected_image,
                caption="Selected Image",
                use_container_width=True
            )

        with col2:
            st.markdown("### Image Information")
            st.write(f"**Resolution:** {width} × {height} pixels")
            st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")

        st.subheader("Classification")

        if st.button(
            "🔍 Classify Image",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Analyzing image... Please wait..."):
                try:
                    results = classify_image(selected_image)

                    best_class_name = results[0][1]
                    proper_object_name = get_proper_object_name(
                        best_class_name
                    )

                    # REAL confidence from MobileNetV2.
                    real_confidence = float(results[0][2]) * 100

                    # Project presentation score requested by user.
                    prediction_score = 100.00

                    st.subheader("Classification Result")

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"### Predicted Object: "
                        f"**{proper_object_name}**"
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric(
                            "Prediction Score",
                            "100.00%"
                        )

                    with result_col2:
                        st.metric(
                            "Model Confidence",
                            f"{real_confidence:.2f}%"
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

                    st.subheader("Top 5 Predictions")

                    for i, (class_id, class_name, probability) in enumerate(
                        results, start=1
                    ):
                        percentage = float(probability) * 100
                        display_name = get_proper_object_name(class_name)

                        st.markdown(
                            f'<div class="prediction-card">'
                            f'<b>{i}. {display_name}</b> — '
                            f'{percentage:.2f}%'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                        st.progress(
                            min(max(float(probability), 0.0), 1.0)
                        )

                    st.subheader("Model Information")

                    info_col1, info_col2, info_col3 = st.columns(3)

                    with info_col1:
                        st.write("**Model**")
                        st.write("MobileNetV2")

                    with info_col2:
                        st.write("**Framework**")
                        st.write("TensorFlow")

                    with info_col3:
                        st.write("**Dataset**")
                        st.write("ImageNet")

                except Exception as error:
                    st.error(f"Classification Error: {error}")

    except Exception as error:
        st.error(f"Image Error: {error}")


st.markdown(
    '<div class="footer">'
    'BCA Minor Project | TensorFlow + MobileNetV2 + Streamlit'
    '</div>',
    unsafe_allow_html=True
)
