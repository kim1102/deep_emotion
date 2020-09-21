"""
Written by dev-kim
kim1102@kist.re.kr
2020.09.20
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision.transforms import transforms

from face_recognition.model_utils import face_verifier
from face_recognition.mtcnn_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face
from resnet import resnet50

class face_box():
    def __init__(self, x1, y1, x2, y2, feat, current_frame, emotion_predict):
        self.emotion = emotion_predict  # neutral
        self.neut = 0
        self.smile = 0
        # coordinate
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        # other option
        self.tt = 0
        self.feat = feat
        self.center = np.array([int((x1 + x2) / 2), int((y1 + y2) / 2)])
        self.frame = current_frame

    def update_state(self, x1, y1, x2, y2, label, current_frame):
        self.x1 = int((self.x1 + x1) / 2)
        self.x2 = int((self.x2 + x2) / 2)
        self.y1 = int((self.y1 + y1) / 2)
        self.y2 = int((self.y2 + y2) / 2)
        self.center = np.array([int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)])
        # emotion
        self.neut += label[0]
        self.smile += label[1]
        self.emotion = np.argmax([self.neut, self.smile])

        self.tt += 1
        if self.tt % 90 == 0: # confidence update
            self.neut /= 2
            self.smile /= 2

        self.frame = current_frame

def check_same_person(tracking_faces, face_model, x1, y1, x2, y2, feat, emotion_label, current_frame, emotion_predict):
    try:
        for idx in range(len(tracking_faces)):
            face = tracking_faces[idx]
            box_center = np.array([int((x1 + x2) / 2), int((y1 + y2) / 2)])  # [x1, y1, x2, y2]
            box_distance = np.linalg.norm(box_center-face.center)
            face_bool = face_model.verify_person(face.feat, feat)
            if face_bool == 0:
                tracking_faces[idx].update_state(x1,y1,x2,y2, emotion_label, current_frame)
                return tracking_faces
            elif face_bool == 1 and box_distance < 20:
                tracking_faces[idx].update_state(x1,y1,x2,y2, emotion_label, current_frame)
                return tracking_faces
            else:
                continue
    except:
        pass
    tracking_faces.append(face_box(x1, y1, x2, y2, feat, current_frame, emotion_predict))
    return tracking_faces  # no same box

def video_smile(video_path):
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    face_reference = get_reference_facial_points(default_square= True)

    # emotion model
    emotion_pretrained = "./trained_model/45.pt"
    emotion_model = resnet50(pretrained=True)
    emotion_model.fc = nn.Linear(2048, 2)
    emotion_model.cuda()
    emotion_model.load_state_dict(torch.load(emotion_pretrained))
    emotion_model.eval()

    # create mtcnn model
    mtcnn = MTCNN(select_largest=False, post_process=False, device="cuda:0")

    # face recognition model
    face_model = face_verifier()

    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # DIVX CODEC
    out = cv2.VideoWriter("./result.avi", fcc, 30, (960, 540))
    cap = cv2.VideoCapture(video_path)

    frame_counter = 0
    tracking_faces = []

    while (cap.isOpened()):
        try:
            ret, frame = cap.read()
            # resize to half
            resized_frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            bboxes, confidence, landmarks = mtcnn.detect(resized_frame, landmarks=True)
        except:
            break

        frame_counter += 1
        faces = []
        emotion_labels = []
        emotion_predicts = []
        try:
            # for debug
            for idx in range(len(bboxes)):
                bbox = bboxes[idx]
                bbox = [int(x) for x in bbox]
                face_img = resized_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                tensor_img = transformations(face_img).unsqueeze(0)
                emotion_output = emotion_model(tensor_img.cuda())
                _, prediction = torch.max(emotion_output, dim=1)
                emotion_predict = prediction.cpu().numpy()[0]

                # Cropped face
                facial5points = landmarks[idx]
                warped_face = warp_and_crop_face \
                    (np.array(resized_frame), facial5points, face_reference, crop_size=(112, 112))
                #cv2.imshow("face_cropped", warped_face)
                #cv2.waitKey(3)

                emotion_predicts.append(emotion_predict)
                emotion_labels.append(emotion_output.data.cpu().numpy()[0])
                faces.append(transformations(warped_face).unsqueeze(0))
        except:
            print("Log: No face detected")
            pass

        # update all tracking boxes
        try:
            for idx in range(len(bboxes)):
                bbox = bboxes[idx]
                bbox = [int(x) for x in bbox]
                emotion_label = emotion_labels[idx]
                emotion_predict = emotion_predicts[idx]
                feat = faces[idx]
                tracking_faces = check_same_person\
                    (tracking_faces, face_model, bbox[0], bbox[1], bbox[2], bbox[3],
                     feat, emotion_label, frame_counter, emotion_predict)
        except:
            print("Log: Cannot update the tracking faces")
            pass

        # delete all unupdated boxes
        try:
            for idx in range(len(tracking_faces)):
                if tracking_faces[idx].frame < frame_counter-3: # keep 3 frame for face tracking
                    tracking_faces.pop(idx)
        except:
            print("Log: Erase all unupdate the tracking faces")
            pass

        # draw all for current tracking faces
        for box in tracking_faces:
            if box.frame == frame_counter: # draw only for current detected box
                if box.emotion == 0:  # neutral
                    text = "neutral"
                    box_color = (255, 255, 255)
                elif box.emotion == 1:  # smile
                    text = "smile"
                    box_color = (0, 0, 255)
                cv2.putText(resized_frame, text, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color)
                cv2.rectangle(resized_frame, (box.x1, box.y1), (box.x2, box.y2), box_color, 3)

        cv2.imshow('result', resized_frame)
        out.write(resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "./test_video2.mp4"
    video_smile(video_path)