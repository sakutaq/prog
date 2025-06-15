import cv2
import os
import glob
import numpy as np


def highlight_face(net, frame, threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()

    face_boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            face_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(h / 150)), 8)
    return frame, face_boxes


def resize_image(img, max_width=800, max_height=600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def find_images(extensions=('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp')):
    files = []
    for ext in extensions:
        files += glob.glob(f'*.{ext}')
    return sorted(files)


def process_image_mode(net):
    images = find_images()
    if not images:
        print("Нет изображений в текущей директории.")
        return

    print("\nДоступные изображения:")
    for i, fname in enumerate(images, 1):
        print(f"{i}: {fname}")

    try:
        idx = int(input("\nВведите номер изображения: ")) - 1
        if not 0 <= idx < len(images):
            raise ValueError
    except ValueError:
        print("Некорректный выбор.")
        return

    path = images[idx]
    img = cv2.imread(path)
    if img is None:
        print(f"Не удалось загрузить изображение: {path}")
        return

    result, boxes = highlight_face(net, img)
    print(f"Лиц найдено: {len(boxes)}" if boxes else "Лица не найдены.")
    cv2.imshow(f"Результат: {os.path.basename(path)}", resize_image(result))
    cv2.waitKey(0)


def process_camera_mode(net):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера недоступна.")
        return

    print("Режим камеры. Нажмите 'q' для выхода.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, _ = highlight_face(net, frame)
        cv2.imshow("Распознавание лиц (камера)", resize_image(result))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    face_model = "opencv_face_detector_uint8.pb"
    face_proto = "opencv_face_detector.pbtxt"

    if not (os.path.exists(face_model) and os.path.exists(face_proto)):
        print("Файлы модели не найдены.")
        return

    net = cv2.dnn.readNet(face_model, face_proto)

    print("\nВыберите режим:")
    print("1 — Камера")
    print("2 — Изображение из папки")

    choice = input("Ваш выбор (1/2): ").strip()
    if choice == '1':
        process_camera_mode(net)
    elif choice == '2':
        process_image_mode(net)
    else:
        print("Некорректный выбор.")

    print("Готово.")

if __name__ == "__main__":
    main()