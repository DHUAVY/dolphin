from modelscope.utils.cv.image_utils import show_video_tracking_result
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from utils import generate_video_name_mp4

class OSTrack:
    
    def __init__(self) -> None:
        pass
    
    def inference(self, inputs):
        splits = inputs.split(",")
        video_path = splits[0]
        track_video_name = generate_video_name_mp4().split('.')[0]
        track_video_name += '.avi'
        
        x1, y1, x2, y2 = int(splits[1]), int(splits[2]), int(splits[3]), int(splits[4])
        init_bbox = [x1, y1, x2, y2]
        
        video_single_object_tracking = pipeline(
            Tasks.video_single_object_tracking, 
            model='damo/cv_vitb_video-single-object-tracking_ostrack')
        
        result = video_single_object_tracking((video_path, init_bbox))
        show_video_tracking_result(video_path, result[OutputKeys.BOXES], track_video_name)
        
        return track_video_name
    