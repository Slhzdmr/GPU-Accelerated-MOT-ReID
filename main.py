import cv2
import torch

from reid_lru_gallery import LRUReIDGallery
from vision_engine import VisionEngine


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] System hardware initialized: {device}")

    vision = VisionEngine(device) #starting the modules as class
    gallery = LRUReIDGallery(max_identities=50, similarity_threshold=0.85, device=device)

    cap = cv2.VideoCapture(0) #start the camera
    if not cap.isOpened():
        print("[ERROR] Failed to start the camera.")
        return

    print("[INFO] Starting, press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret: #is frame returned?
            break

        valid_data, embeddings = vision.extract_features(frame)

        if embeddings is not None:
            for i in range(len(valid_data)):
                track_id = valid_data[i]["id"]
                x1, y1, x2, y2 = valid_data[i]["box"]

                current_embedding = embeddings[i]
                matched_id, similarity_score = gallery.match_identity(current_embedding)

                if matched_id is not None:
                    final_label = f"ReID: {matched_id} ({similarity_score:.2f})"
                    box_color = (0, 255, 0)
                else:
                    new_id = f"Person_{int(track_id)}"
                    gallery.register_or_update(new_id, current_embedding)
                    final_label = f"NEW: {new_id}"
                    box_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.imshow("MOT and ReID_v1", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()