from pathlib import Path
import sys

class DetectionSettings:
    """
    A class for managing detection settings.

    Attributes:
        file_path (Path): Absolute path of the current file.
        root_path (Path): Parent directory of the current file.
        ROOT (Path): Relative path of the root directory with respect to the current working directory.
        IMAGE (str): Constant representing image source.
        VIDEO (str): Constant representing video source.
        DRONE (str): Constant representing drone camera source.
        SOURCES_LIST (list): List of available sources.
        IMAGES_DIR (Path): Directory for storing images.
        DEFAULT_IMAGE (Path): Default image path.
        DEFAULT_DETECT_IMAGE (Path): Default path for the image with detection overlay.
        VIDEO_DIR (Path): Directory for storing videos.
        MODEL_DIR (Path): Directory for storing ML models.
        DETECTION_MODEL (Path): Path to the detection model file.
        WEBCAM_PATH (int/str): Path to the webcam (0 for default webcam).
        custom_color (tuple): Custom color in (B, G, R) format for drawing rectangles.
        class_ids (list): List of class IDs for filtering detections.
    """

    def __init__(self):
        """
        Initialize detection settings.
        """
        # For Getting the absolute path of the current file
        self.file_path = Path(__file__).resolve()

        # For Getting the parent directory of the current file
        self.root_path = self.file_path.parent

        # Add the root path to the sys.path list if it is not already there
        if self.root_path not in sys.path:
            sys.path.append(str(self.root_path))

        # For Getting the relative path of the root directory with respect to the current working directory
        self.ROOT = self.root_path.relative_to(Path.cwd())

        # Sources
        self.IMAGE = 'Image'
        self.VIDEO = 'Video'
        self.DRONE = 'Drone Cam'

        self.SOURCES_LIST = [self.IMAGE, self.VIDEO, self.DRONE]

        # Images config
        self.IMAGES_DIR = self.ROOT / 'images'
        self.DEFAULT_IMAGE = self.IMAGES_DIR / 'test.jpg'
        self.DEFAULT_DETECT_IMAGE = self.IMAGES_DIR / 'test_detat.jpg'

        # Videos config
        self.VIDEO_DIR = self.ROOT / 'videos'

        # ML Model config
        self.MODEL_DIR = self.ROOT / 'weights'
        self.DETECTION_MODEL = self.MODEL_DIR / 'yolov8n.pt'

        # Webcam
        self.WEBCAM_PATH = 0

        # Specify custom color in (B, G, R)
        self.COUSTOM_COLOR = (128, 0, 128)  

        # Initialize id for class
        self.CLASS_IDS = [0]
