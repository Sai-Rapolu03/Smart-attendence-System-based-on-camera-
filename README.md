# Face Recognition-Based Attendance System using ESP32-CAM and OpenCV

## 📌 Project Overview

This project presents a **real-time face recognition-based attendance system** developed using the **ESP32-CAM module** and **OpenCV**. It addresses the inefficiencies of traditional attendance methods such as manual sign-ins or card swipes, which are time-consuming and prone to fraud (e.g., proxy attendance).

By combining embedded hardware with computer vision, the system offers an **automated, contactless, and reliable attendance solution** for classrooms, offices, or any environment where accurate presence tracking is required.

## 🧠 Key Features

- 🎥 Real-time video streaming via ESP32-CAM
- 🧑‍💼 Face detection and recognition using **LBPH (Local Binary Patterns Histograms)**
- ✅ High-accuracy recognition based on a pre-trained dataset
- 📊 Attendance logging to **Excel sheets** with name, roll number, date, and timestamp
- 📁 Monthly report generation
- 🧼 Fully contactless and hygienic
- 💸 Low-cost and scalable solution

## 🛠️ Technologies Used

- **ESP32-CAM** (Microcontroller with onboard camera and Wi-Fi)
- **OpenCV** (for face detection & recognition)
- **Python** (for face recognition and attendance logging)
- **Excel (CSV/XLSX)** (for attendance data storage)
- **Flask** *(optional)* for local web interface

## 🔧 System Architecture

1. **ESP32-CAM** captures and streams live video.
2. Video stream is processed by **OpenCV** on the host PC.
3. Faces are detected and matched with the pre-trained dataset using **LBPH**.
4. If a face is recognized:
   - Attendance is recorded with name, roll number, timestamp.
   - Data is saved in a monthly **Excel sheet**.
   



