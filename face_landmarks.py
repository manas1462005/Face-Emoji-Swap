import cv2
import mediapipe as mp
import numpy as np

# Load emojis with alpha channel (transparent background)
emoji_smile = cv2.imread('emojis/emoji_smile.png', cv2.IMREAD_UNCHANGED)
emoji_surprised = cv2.imread('emojis/emoji_surprised.png', cv2.IMREAD_UNCHANGED)
emoji_neutral = cv2.imread('emojis/emoji_neutral.png', cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Helper functions
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Overlay emoji with transparency on the frame at (x, y) resized to (width, height)
def overlay_emoji(frame, emoji_img, x, y, width, height):
    # Resize emoji to face bounding box size
    emoji_resized = cv2.resize(emoji_img, (width, height), interpolation=cv2.INTER_AREA)
    
    if emoji_resized.shape[2] == 4:
        emoji_rgb = emoji_resized[:, :, :3]
        alpha_mask = emoji_resized[:, :, 3] / 255.0
    else:
        emoji_rgb = emoji_resized
        alpha_mask = np.ones(emoji_rgb.shape[:2], dtype=float)
    
    h, w = emoji_rgb.shape[:2]
    rows, cols, _ = frame.shape

    # Calculate ROI on the frame
    x1, x2 = max(0, x), min(cols, x + w)
    y1, y2 = max(0, y), min(rows, y + h)

    # Corresponding region on emoji
    emoji_x1, emoji_x2 = 0, x2 - x1
    emoji_y1, emoji_y2 = 0, y2 - y1

    if x1 >= x2 or y1 >= y2:
        return frame  # Out of bounds

    roi = frame[y1:y2, x1:x2]
    emoji_region = emoji_rgb[emoji_y1:emoji_y2, emoji_x1:emoji_x2]
    alpha_region = alpha_mask[emoji_y1:emoji_y2, emoji_x1:emoji_x2]

    for c in range(3):
        roi[:, :, c] = (alpha_region * emoji_region[:, :, c] +
                        (1 - alpha_region) * roi[:, :, c])

    frame[y1:y2, x1:x2] = roi
    return frame

# Mouth landmarks indices from MediaPipe Face Mesh
MOUTH_LANDMARKS = [61, 291, 78, 308, 13, 14, 312, 82]

# For MAR calculation (mouth aspect ratio)
MAR_VERTICAL_PAIRS = [(13, 14), (78, 308), (82, 312)]
MAR_HORIZONTAL = (61, 291)

# Simple smoothing function for landmarks across frames
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.last = None

    def smooth(self, new):
        if self.last is None:
            self.last = new
            return new
        smoothed = self.alpha * np.array(new) + (1 - self.alpha) * np.array(self.last)
        self.last = smoothed
        return smoothed.astype(int)

smoothers = {idx: LandmarkSmoother() for idx in MOUTH_LANDMARKS}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_detection = face_detection.process(rgb_frame)
    results_mesh = face_mesh.process(rgb_frame)

    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw * 1.5)
            h = int(bboxC.height * ih * 1.5)
            # Draw bbox (optional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if results_mesh.multi_face_landmarks:
        for face_landmarks in results_mesh.multi_face_landmarks:
            ih, iw, _ = frame.shape

            mouth_points = []
            for idx in MOUTH_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                px = int(lm.x * iw)
                py = int(lm.y * ih)
                smoothed_point = smoothers[idx].smooth((px, py))
                mouth_points.append(smoothed_point)
                cv2.circle(frame, tuple(smoothed_point), 3, (255, 0, 0), -1)

            left_corner = mouth_points[MOUTH_LANDMARKS.index(MAR_HORIZONTAL[0])]
            right_corner = mouth_points[MOUTH_LANDMARKS.index(MAR_HORIZONTAL[1])]
            mouth_width = euclidean(left_corner, right_corner)

            vertical_distances = []
            for (upper_idx, lower_idx) in MAR_VERTICAL_PAIRS:
                upper = mouth_points[MOUTH_LANDMARKS.index(upper_idx)]
                lower = mouth_points[MOUTH_LANDMARKS.index(lower_idx)]
                vertical_distances.append(euclidean(upper, lower))
            mouth_height = np.mean(vertical_distances)

            ratio = mouth_height / mouth_width if mouth_width != 0 else 0
            print(f"MAR (mouth openness ratio): {ratio:.2f}")

            # Decide expression
            if ratio > 0.55:
                expression = "😮 Surprised"
                emoji_to_show = emoji_surprised
            elif ratio > 0.42:
                expression = "😀 Smiling"
                emoji_to_show = emoji_smile
            else:
                expression = "😐 Neutral"
                emoji_to_show = emoji_neutral

            # Display text near mouth left corner
            cv2.putText(frame, expression, (left_corner[0], left_corner[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Overlay emoji covering whole face bbox
            if results_detection.detections:
                # Use first detection bbox for overlay position
                detection = results_detection.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                frame = overlay_emoji(frame, emoji_to_show, x, y, w, h)

    cv2.imshow("MediaPipe Face Detection + Face Mesh Mouth Expression", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
