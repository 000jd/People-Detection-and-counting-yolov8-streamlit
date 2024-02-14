from ultralytics import YOLO
import streamlit as st
import cv2
import utils.settings as settings_module
import numpy as np
import supervision as sv

class DetectionHelper:
    def __init__(self):
        # Initialize settings and helper modules
        self.settings = settings_module.DetectionSettings()

        # Variable for storing total detactions
        self.total_detections = 0

    def load_model(self, model_name, model_path):
        """
        Loads a YOLO object detection model from the specified model_path.

        Parameters:
            model_name (str): Name of the model.
            model_path (str): The path to the YOLO model file.

        Returns:
            A YOLO object detection model.
        """
        try:
            model = YOLO(model_path)
            #st.success(f"Model '{model_name}' loaded successfully!")
            return model
        except Exception as ex:
            st.error(f"Unable to load model '{model_name}'. Check the specified path: {model_path}")
            st.error(ex)
            return None
    
    def filter_detection(self, result, class_ids):
        """
        Filter detections for a particular class.

        Parameters:
            result (dict): Dictionary containing YOLOv8 detection results.
            class_id (int): ID of the class to filter detections for.

        Returns:
            dict: Filtered detection results.
        """
        # Convert YOLOv8 detection results to Supervision Detections object
        detections = sv.Detections.from_ultralytics(result[0])

        # Filter detections for the specified class
        filtered_detections = detections[np.isin(detections.class_id, class_ids)]

        return filtered_detections

    def display_tracker_options(self):
        is_display_tracker = True
        if is_display_tracker:
            tracker_type = "bytetrack.yaml"
            return is_display_tracker, tracker_type
        return is_display_tracker, None

    def _display_detected_frames(self, conf, model, st_frame, image, is_display_tracking=None, tracker=None):
        """
        Display the detected objects on a video frame using the YOLOv8 model.

        Args:
        - conf (float): Confidence threshold for object detection.
        - model (YoloV8): A YOLOv8 object detection model.
        - st_frame (Streamlit object): A Streamlit object to display the detected video.
        - image (numpy array): A numpy array representing the video frame.
        - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

        Returns:
        None
        """

        # Resize the image to a standard size
        image_resized = cv2.resize(image, (720, int(720*(9/16))))

        # Display object tracking, if specified
        if is_display_tracking:
            res = model.track(image, conf=conf, persist=True, tracker=tracker)
        else:
            # Predict the objects in the image using the YOLOv8 model
            res = model.predict(image, conf=conf)
        # Filter detections for class ID 0
        filtered_result = self.filter_detection(res, class_ids=self.settings.CLASS_IDS)

        # counts the number of detactions
        self.total_detections = len(filtered_result)
        
        # Draw filtered detections on the image
        for detection in filtered_result.xyxy:
            x1, y1, x2, y2 = map(int, detection)
            cv2.rectangle(image_resized, (x1, y1), (x2, y2), self.settings.COUSTOM_COLOR, thickness=self.settings.THICKNESS)
            
        st_frame.image(image_resized,
                    caption=f'Detected Video (Total Detections: {self.total_detections})',
                    channels="BGR",
                    use_column_width=True
                    )

    def drone_camera_classification(self, conf, model):
        """
        Detects objects in real-time using the YOLOv8 object detection model with a drone camera.

        Parameters:
            conf: Confidence of YOLOv8 model.
            model: An instance of the `YOLOv8` class containing the YOLOv8 model.

        Returns:
            None
        """
        source = st.sidebar.radio("Select Source", ("IP Cam", "Webcam"))

        if source == "IP Cam":
            ip_cam_url = st.sidebar.text_input("Enter IP Cam URL:")
            if not ip_cam_url:
                st.sidebar.info("Please enter the IP Cam URL.")
                return
            
            # Functionality for IP Cam
            try:
                vid_cap = cv2.VideoCapture(ip_cam_url)
                st_frame = st.empty()
                is_display_tracker, tracker = self.display_tracker_options()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        self._display_detected_frames(conf,
                                                    model,
                                                    st_frame,
                                                    image,
                                                    is_display_tracker,
                                                    tracker,
                                                    )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
                
        elif source == "Webcam":
            st.sidebar.info("Please connect webcam befor use")
            # Display dropdown to select webcam path
            selected_webcam = st.sidebar.selectbox("Select Webcam", self.settings.WEBCAM_PATH)
            source_webcam = selected_webcam
            is_display_tracker, tracker = self.display_tracker_options()
            if st.sidebar.button('Detect Objects'):
                try:
                    vid_cap = cv2.VideoCapture(source_webcam)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            self._display_detected_frames(conf,
                                                        model,
                                                        st_frame,
                                                        image,
                                                        is_display_tracker,
                                                        tracker,
                                                        )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.sidebar.error("Error loading video: " + str(e))
        else:
            st.sidebar.error("Error processing video Source") 

    def play_video(self, conf, model, video_path):
        vid_cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                self._display_detected_frames(conf, model, st_frame, image)
            else:
                vid_cap.release()
                break

    def video_classification(self, conf, model):
        uploaded_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            video_path = f"uploaded_video.{uploaded_file.name.split('.')[-1]}"
            with open(video_path, 'wb') as video_file:
                video_file.write(uploaded_file.read())
        else:
            st.sidebar.info("Please upload a video file.")
            return

        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)

        if st.sidebar.button('Detect Video Objects'):
            try:
                vid_cap = cv2.VideoCapture(video_path)
                st_frame = st.empty()
                
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        # Resize the image to a standard size
                        image_resized = cv2.resize(image, (720, int(720*(9/16))))

                        # Display the detected objects on the video frame
                        res = model.predict(image_resized, conf=conf)
                        # Filter detections for class ID 0
                        filtered_result = self.filter_detection(res, class_ids=self.settings.CLASS_IDS)

                        # counts the number of detactions
                        self.total_detections = len(filtered_result)

                        # Draw filtered detections on the image
                        for detection in filtered_result.xyxy:
                            x1, y1, x2, y2 = map(int, detection)
                            cv2.rectangle(image_resized, (x1, y1), (x2, y2), self.settings.COUSTOM_COLOR, thickness=self.settings.THICKNESS)

                        #res_plotted = filtered_result[0].plot()
                        st_frame.image(image_resized, caption=f'Detected Video (Total Detections: {self.total_detections})', channels="BGR", use_column_width=True)

                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                vid_cap.release()
                st.sidebar.error("Error processing video: " + str(e))


