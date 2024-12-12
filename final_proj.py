
import os
import tensorflow as tf
import streamlit as st
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


devices = tf.config.list_physical_devices()
st.write("Using device(s):", devices)


def load_model_with_custom_objects(model_path):
    from tensorflow.keras.losses import SparseCategoricalCrossentropy

    def custom_sparse_categorical_crossentropy(**kwargs):
        """Handle unexpected arguments in deserialization."""
        kwargs.pop("fn", None)  # Remove unsupported 'fn' key if present
        return SparseCategoricalCrossentropy(**kwargs)

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "SparseCategoricalCrossentropy": custom_sparse_categorical_crossentropy
            },
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model_with_custom_objects("cifar10_model.h5")


st.title("Image Classification App")

def label_class(class_number):
    """Maps class indices to human-readable labels."""
    class_labels = {
        0: "Airplane",
        1: "Automobile",
        2: "Bird",
        3: "Cat",
        4: "Deer",
        5: "Dog",
        6: "Frog",
        7: "Horse",
        8: "Ship",
        9: "Truck",
    }
    return class_labels.get(class_number, "Invalid class number")


with st.sidebar:
    st.header("Upload Image")
    st.caption(
        "This app is limited to classifying the following categories: airplane, "
        "automobile, bird, cat, deer, dog, frog, horse, ship, and truck."
    )
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
      
        image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(32, 32))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)  # Create a batch

        image_array /= 255.0

      
        predictions = model.predict(image_array)
        top_classes = tf.argsort(predictions[0], direction="DESCENDING")[:3]  # Top 3 predictions
        top_scores = tf.nn.softmax(predictions[0][top_classes])

     
        highest_confidence_idx = top_classes[0].numpy()
        highest_confidence_label = label_class(highest_confidence_idx)
        highest_confidence_score = 100 * top_scores[0].numpy()
        st.write(f"Highest Confidence: {highest_confidence_label} ({highest_confidence_score:.2f}%)")

      
        st.subheader("Top Predictions")
        for i, (class_idx, score) in enumerate(zip(top_classes.numpy(), top_scores.numpy())):
            class_label = label_class(class_idx)
            st.write(f"{i + 1}. {class_label}: {100 * score:.2f}%")

       
        st.image(image, caption="Uploaded Image", use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
else:
    st.info("Please upload an image to proceed.")
