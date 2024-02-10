from pathlib import Path
import PIL
import streamlit as st
import utils.settings as settings
import utils.helper as helper

class AccidentDetectionApp:
    def __init__(self):
        st.set_page_config(
            page_title="Accident Detection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.model = None
        self.confidence = None
        self.source_radio = None
        self.source_img = None

    def load_model(self, model_path):
        try:
            self.model = helper.load_model(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

    def show_detection_page(self):
        st.title("Accident Detection")

        # Sidebar
        st.sidebar.header("Navigation")

        # Sidebar options
        self.confidence = float(st.sidebar.slider(
            "Select Model Confidence", 25, 100, 40)) / 100

        self.source_radio = st.sidebar.radio(
            "Select Source", settings.SOURCES_LIST)

        # Main content for Detection
        if self.source_radio == settings.IMAGE:
            self.source_img = st.sidebar.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

            col1, col2 = st.columns(2)

            with col1:
                try:
                    if self.source_img is None:
                        default_image_path = str(settings.DEFAULT_IMAGE)
                        default_image = PIL.Image.open(default_image_path)
                        st.image(default_image_path, caption="Default Image",
                                use_column_width=True)
                    else:
                        uploaded_image = PIL.Image.open(self.source_img)
                        st.image(self.source_img, caption="Uploaded Image",
                                use_column_width=True)
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if self.source_img is None:
                    default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                    default_detected_image = PIL.Image.open(
                        default_detected_image_path)
                    st.image(default_detected_image_path, caption='Detected Image',
                            use_column_width=True)
                else:
                    if st.sidebar.button('Detect Objects'):
                        res = self.model.predict(uploaded_image,
                                                conf=self.confidence
                                                )
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image',
                                use_column_width=True)

                        try:
                            with st.expander("Detection Results"):
                                for box in boxes:
                                    st.write(box.data)
                        except Exception as ex:
                            st.write("No image is uploaded yet!")

        elif self.source_radio == settings.VIDEO:
            helper.video_clsifiction(self.confidence, self.model)

        elif self.source_radio == settings.DRONE:
            # Functionality for Drone Camera
            pass

        else:
            st.error("Please select a valid source type!")

    def run(self):
        self.load_model(Path(settings.DETECTION_MODEL))
        self.show_detection_page()

app = AccidentDetectionApp()
app.run()
