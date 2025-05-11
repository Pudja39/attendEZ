import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def daftar_wajah():
    cam = cv2.VideoCapture(0)
    face_id = input("Masukkan NIM: ").strip()

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    existing = [f for f in os.listdir('dataset') if f"User.{face_id}." in f]
    if existing:
        print(f"Wajah dengan NIM {face_id} sudah ada.")
        cam.release()
        return

    print("Arahkan wajah ke kamera. Ambil 20 gambar dengan berbagai ekspresi dan posisi...")
    count = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            filename = f"dataset/User.{face_id}.{count}.jpg"
            cv2.imwrite(filename, face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Gambar {count}/20", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('Pendaftaran Wajah', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Pendaftaran wajah selesai.")

def train_model():
    path = 'dataset'
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        gray_img = Image.open(image_path).convert('L')
        image_np = np.array(gray_img, 'uint8')
        try:
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
        except ValueError:
            continue
        faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y+h, x:x+w])
            ids.append(face_id)

    if not face_samples:
        print("Tidak ada data wajah yang terdeteksi.")
        return

    recognizer.train(face_samples, np.array(ids))
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    recognizer.write('trainer/trainer.yml')
    print("Model pelatihan berhasil disimpan.")

def absensi():
    if not os.path.exists('trainer/trainer.yml'):
        print("Model tidak ditemukan. Silakan latih terlebih dahulu.")
        return

    recognizer.read('trainer/trainer.yml')
    cam = cv2.VideoCapture(0)
    absen_file = 'attend.csv'
    recognized_ids = []
    confidence_threshold = 70  # boleh disesuaikan

    print("Mulai absensi. Arahkan wajah ke kamera...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Gagal membaca dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            id_predicted, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < confidence_threshold:
                if id_predicted not in recognized_ids:
                    tanggal = datetime.now().strftime('%Y-%m-%d')
                    jam = datetime.now().strftime('%H:%M:%S')
                    df = pd.DataFrame([{'NIM': id_predicted, 'Tanggal': tanggal, 'Jam': jam}])
                    if os.path.exists(absen_file):
                        df.to_csv(absen_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(absen_file, index=False)
                    recognized_ids.append(id_predicted)
                    print(f"[✔] NIM {id_predicted} tercatat hadir.")

                    # Tampilkan wajah dan label sebentar lalu keluar
                    label = f"NIM: {id_predicted}"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow('Absensi Wajah', frame)
                    cv2.waitKey(1500)  # tampilkan 1.5 detik
                    cam.release()
                    cv2.destroyAllWindows()
                    return  # keluar dari fungsi setelah absensi berhasil

            else:
                print("[✘] Wajah tidak dikenali.")
                label = "Tidak dikenal"
                color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Absensi Wajah', frame)

        # Kamera tetap aktif jika belum ada wajah yang dikenali
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("\nAttendEZ - Menu")
        print("1. Daftar Wajah")
        print("2. Latih Model")
        print("3. Mulai Absensi")
        print("4. Keluar")
        pilihan = input("Pilih opsi (1-4): ").strip()

        if pilihan == '1':
            daftar_wajah()
        elif pilihan == '2':
            train_model()
        elif pilihan == '3':
            absensi()
        elif pilihan == '4':
            print("Keluar dari program.")
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")

if __name__ == "__main__":
    main()
