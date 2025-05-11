AttendEZ - Sistem Absensi Mahasiswa Berbasis Pengenalan Wajah
==============================================================

Deskripsi:
----------
AttendEZ adalah aplikasi absensi otomatis berbasis pengenalan wajah menggunakan OpenCV dan Python.
Aplikasi ini mendeteksi wajah mahasiswa dan mencatat kehadiran secara otomatis ke dalam file CSV.

Struktur Direktori:
-------------------
├── AttendEZ.py             -> Skrip utama
├── dataset/                -> Folder penyimpanan gambar wajah hasil pendaftaran
├── trainer/trainer.yml     -> File model hasil pelatihan pengenalan wajah
├── attend.csv              -> File absensi (otomatis dibuat)
├── mahasiswa.csv (opsional)-> Data NIM dan Nama mahasiswa (bisa ditambahkan)
├── requirements.txt        -> Daftar dependensi

Fitur:
------
1. Daftar Wajah:
   - Memasukkan NIM mahasiswa
   - Mengambil 20 gambar wajah dari webcam
   - Menyimpan gambar ke folder dataset/

2. Latih Model:
   - Melatih model LBPH Face Recognizer berdasarkan dataset
   - Menyimpan model ke trainer/trainer.yml

3. Mulai Absensi:
   - Menyalakan webcam
   - Mendeteksi wajah dan mencocokkannya dengan model
   - Jika cocok, mencatat NIM dan waktu kehadiran ke attend.csv
   - Kamera akan otomatis tertutup setelah 1 wajah berhasil dicatat

4. Keluar:
   - Menutup aplikasi dari menu utama

Cara Menjalankan:
-----------------
1. Buka terminal/command prompt di direktori AttendEZ.
2. Install dependensi:
   pip install -r requirements.txt

3. Jalankan program:
   python AttendEZ.py

Catatan Penting:
----------------
- Gunakan pencahayaan yang baik saat pendaftaran wajah.
- Usahakan wajah tidak terlalu jauh atau terlalu dekat ke kamera.
- Jika wajah tidak dikenali, pastikan model telah dilatih dan dataset sudah lengkap.
- Wajah yang berhasil dikenali akan menyebabkan aplikasi absensi langsung tertutup.

Dibuat oleh: [Nama Kamu]
Universitas Pancasakti Tegal
