from pathlib import Path
import sys

class AccidentDetectionSettings:
    def __init__(self):
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


