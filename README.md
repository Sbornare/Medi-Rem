# Voice-enabled Medicine Reminder

## 📌 Project Overview
The **Voice-enabled Medicine Reminder** is a Flask-based AI-powered application designed to help users manage their medication schedules effectively. It extracts medicine details from prescription images using **OCR (Optical Character Recognition)** and schedules reminders accordingly. The system is particularly useful for senior citizens and individuals with complex medication routines.

---

## 🚀 Features
- **Prescription Image Processing:** Extracts medicine details from images using PaddleOCR.
- **Medicine & Dosage Detection:** Identifies medicine names, dosages, and frequencies using regex.
- **Automated Reminders:** Schedules medicine intake reminders based on the extracted information.
- **Flask API Endpoints:** Provides RESTful APIs for image upload and text extraction.
- **Scheduler Integration:** Uses `APScheduler` to trigger reminders at set intervals.
- **Device Compatibility:** Supports CPU and GPU (CUDA-enabled for PyTorch models).

---

## 🏗️ Tech Stack
- **Backend:** Flask
- **Machine Learning:** PyTorch (for AI-based processing)
- **OCR:** PaddleOCR (for text extraction)
- **Scheduler:** APScheduler (for reminders)
- **Image Processing:** PIL (Pillow) and OpenCV

---

## 📦 Installation & Setup

### 🔹 Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- pip (latest version)
- Virtual environment (optional but recommended)

### 🔹 Clone the Repository
```bash
git clone https://github.com/HarshBothara24/MediRem.git
cd MediRem
```

### 🔹 Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run the Flask Application
```bash
python app.py
```

### 🔹 Access the API
Once the server is running, open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 📂 API Endpoints
### 1️⃣ **Home Route**
- **Endpoint:** `/`
- **Method:** GET
- **Description:** Displays the homepage.

### 2️⃣ **Test API Connection**
- **Endpoint:** `/apidata`
- **Method:** GET
- **Description:** Returns a test message to confirm API availability.

### 3️⃣ **Upload Prescription for Analysis**
- **Endpoint:** `/predict`
- **Method:** POST
- **Description:** Accepts an image file or URL and extracts medicine information.
- **Request Body:**
  - `file`: Image file (Multipart Form-Data)
  - OR `image_url`: Direct image URL (JSON/Form-Data)
- **Response:**
```json
{
  "extracted_text": "Detected prescription text",
  "extracted_medicine_data": [
    "Paracetamol 500mg - 2x daily - 5 days"
  ]
}
```

---

## 🛠️ How the System Works
1. **Upload Prescription**: Users upload an image of their prescription.
2. **OCR Processing**: PaddleOCR extracts text from the image.
3. **Regex Parsing**: Extracted text is processed to detect medicine names, dosages, and schedules.
4. **Reminder Scheduling**: Using APScheduler, reminders are set at specified times.
5. **Notification System**: The system prints reminders (extendable to SMS or voice alerts).

---

## 📜 Future Enhancements
✅ Integrate with Google Home & Alexa for voice reminders 🔊  
✅ Add SMS or WhatsApp notification support 📲  
✅ Implement user authentication & medicine history tracking 🔐

---

## 🤝 Contributing
Feel free to contribute! Follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-xyz`)
3. Commit your changes (`git commit -m "Added feature XYZ"`)
4. Push to your branch (`git push origin feature-xyz`)
5. Open a Pull Request 🚀

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
For queries or collaboration, reach out to:  
📧 Email: harshbothara24@gmail.com  
🐦 Twitter: https://x.com/HarshBothara24  
💼 LinkedIn: https://www.linkedin.com/in/harshbothara24/

