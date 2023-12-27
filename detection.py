from ultralytics import YOLO
import supervision as sv
import cv2
import os

def detected_image_generator(image_path, ball_pt_path, player_pt_path, save_path):
    image = cv2.imread(image_path)

    model_ball = YOLO(ball_pt_path)
    result_ball = model_ball(image)[0]
    detections_ball = sv.Detections.from_ultralytics(result_ball)

    model_player = YOLO(player_pt_path)
    result_player = model_player(image)[0]
    detections_player = sv.Detections.from_ultralytics(result_player)

    colorScheme = sv.ColorPalette(colors = [sv.Color(r=222, g=159, b=248), sv.Color(r=179, g=253, b=178), sv.Color(r=231, g=129, b=52)])
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position = sv.Position.TOP_LEFT)
    
    annotated_frame_player = bounding_box_annotator.annotate(
        scene = image.copy(),
        detections = detections_player
    )

    annotated_frame_ball = bounding_box_annotator.annotate(
        scene = annotated_frame_player,
        detections = detections_ball
    )

    annotated_frame_player = label_annotator.annotate(
        scene = annotated_frame_ball,
        detections = detections_player,
        labels = [
            result_player.names[class_id] for class_id in detections_player.class_id
        ]
    )

    annotated_frame_ball = label_annotator.annotate(
        scene = annotated_frame_player,
        detections = detections_ball,
        labels = [
            result_ball.names[class_id] for class_id in detections_ball.class_id
        ]
    )

    cv2.imwrite(save_path + os.path.basename(image_path), annotated_frame_ball)

def get_images(folder_path):
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    ret = []
    for file in file_list:
        if file.lower().endswith(('jpg', 'png', 'gif')):
            file_path = os.path.join(folder_path, file)
            ret.append(file_path)

    return ret

img_dir_path = 'SpoitWeb/static/temp'
detected_dir_path = 'SpoitWeb/static/detected/'

img_file_paths = get_images(img_dir_path)
for img_file_path in img_file_paths:
    detected_image_generator(img_file_path, 'SpoitWeb/static/models/ball.pt', 'SpoitWeb/static/models/players.pt', detected_dir_path)