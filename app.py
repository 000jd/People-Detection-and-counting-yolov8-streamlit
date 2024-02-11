from pathlib import Path
import PIL
import streamlit as st
import utils.settings as settings_module
import utils.helper as helper_module

class DetectionApp:
    def __init__(self):
        # Set Streamlit page configuration
        st.set_page_config(
            page_title="People Detection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize settings and helper modules
        self.settings = settings_module.DetectionSettings()
        self.helper = helper_module.DetectionHelper()

        # Initialize model, confidence, source_radio, and source_img
        self.model = None
        self.confidence = None
        self.source_radio = None
        self.source_img = None

    def load_model(self, model_path):
        """
        Load the YOLO object detection model from the specified model_path.

        Parameters:
            model_path (str): The path to the YOLO model file.

        Returns:
            None
        """
        try:
            self.model = self.helper.load_model(model_path)
        except Exception as ex:
            st.error(
                f"Unable to load model. Check the specified path: {model_path}")
            st.error(ex)

    def show_detection_page(self):
        """
        Display the main content for detection based on user-selected options.

        Returns:
            None
        """
        st.title("People Detection")

        # Sidebar
        st.sidebar.header("Navigation")

        # Sidebar options
        self.confidence = float(st.sidebar.slider(
            "Select Model Confidence", 25, 100, 40)) / 100

        self.source_radio = st.sidebar.radio(
            "Select Source", self.settings.SOURCES_LIST)

        # Main content for Detection
        if self.source_radio == self.settings.IMAGE:
            self.source_img = st.sidebar.file_uploader(
                "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

            col1, col2 = st.columns(2)

            with col1:
                try:
                    if self.source_img is None:
                        default_image_path = str(self.settings.DEFAULT_IMAGE)
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
                    default_detected_image_path = str(self.settings.DEFAULT_DETECT_IMAGE)
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

        elif self.source_radio == self.settings.VIDEO:
            self.helper.video_classification(self.confidence, self.model)

        elif self.source_radio == self.settings.DRONE:
            self.helper.drone_camera_classification(self.confidence, self.model)

        else:
            st.error("Please select a valid source type!")

    def run(self):
        """
        Run the accident detection app by loading the model and displaying the detection page.

        Returns:
            None
        """
        self.load_model(Path(self.settings.DETECTION_MODEL))
        self.show_detection_page()

app = DetectionApp()
app.run()
