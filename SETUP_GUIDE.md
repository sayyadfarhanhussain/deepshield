# 🛡️ DeepShield ADVANCED — Complete Setup Guide
### Final Year Project | All Features Combined

---

## 📁 PROJECT STRUCTURE

```
DeepShield/
│
├── app.py                    ← Advanced Flask backend (ALL features)
├── requirements.txt          ← Libraries to install
│
├── templates/
│   └── index.html            ← Full UI (Single, Batch, Webcam, Heatmap, PDF)
│
└── static/
    ├── uploads/              ← Uploaded images (auto-created)
    ├── heatmaps/             ← GradCAM heatmap images (auto-created)
    └── reports/              ← PDF reports (auto-created)
```

---

## ✨ FEATURES IN THIS VERSION

| Feature | Description |
|---------|-------------|
| **Single Detection** | Upload one image → Real/Fake verdict + confidence bars |
| **Batch Detection** | Upload up to 20 images → Analyze all at once |
| **Webcam** | Live camera capture → Real-time deepfake check |
| **GradCAM Heatmap** | See EXACTLY which regions the AI focuses on |
| **PDF Report** | Download a professional PDF of batch results |

---

## ⚙️ SETUP — Step by Step

### STEP 1: Install Python
- Download from: https://www.python.org/downloads/
- Choose Python **3.10 or 3.11**
- ✅ Check **"Add Python to PATH"** during install
- Verify: Open Terminal → `python --version`

---

### STEP 2: Place Your Files

Create a folder called `DeepShield` and organize like this:
```
DeepShield/
├── app.py
├── requirements.txt
└── templates/
    └── index.html        ← MUST be inside templates folder!
```

---

### STEP 3: Open Terminal in VS Code

Open VS Code → Open the `DeepShield` folder → Press `Ctrl + ~`

---

### STEP 4: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

---

### STEP 5: Install All Libraries

```bash
pip install -r requirements.txt
```

⏳ PyTorch is large (~700MB). Wait patiently!

**If PyTorch fails, run this instead:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask pillow numpy opencv-python reportlab werkzeug
```

---

### STEP 6: Run the Server

```bash
python app.py
```

You should see:
```
Loading model...
Model ready!
======================================================
  DeepShield ADVANCED is RUNNING!
  Open: http://127.0.0.1:5000
  Features: Single | Batch | Webcam | Heatmap | PDF
======================================================
```

---

### STEP 7: Open Browser

Go to: **http://127.0.0.1:5000**

You'll see 4 tabs: Single Image · Batch Detect · Webcam · Heatmap 🎉

---

## 🐛 TROUBLESHOOTING

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: flask` | `pip install flask` |
| `ModuleNotFoundError: torch` | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: reportlab` | `pip install reportlab` |
| Port 5000 in use | Change port: `app.run(port=8080)` in app.py |
| `templates/index.html not found` | Make sure index.html is inside a `templates/` folder |
| Webcam not working | Allow camera permission in browser |
| Heatmap returns None | Normal if GradCAM fails on some images |

---

## 📚 LIBRARIES USED (Know for Viva!)

| Library | Purpose |
|---------|---------|
| **Flask** | Web server framework |
| **PyTorch** | Deep learning — runs the AI model |
| **TorchVision** | EfficientNet-B0 model + image transforms |
| **Pillow** | Image file reading/processing |
| **OpenCV** | GradCAM heatmap overlay generation |
| **NumPy** | Array operations for heatmap math |
| **ReportLab** | PDF report generation |
| **Werkzeug** | Secure file upload handling |

---

## 🎤 JUDGE PITCH (Say This!)

> "My project DeepShield is an AI-powered deepfake detection system with 5 advanced features.
>
> It uses **EfficientNet-B0** with transfer learning to classify images as Real or Fake.
> The **GradCAM heatmap** shows exactly which pixels the AI focused on — making it fully explainable.
> The **batch mode** processes 20 images at once, and results can be exported as a **professional PDF report**.
> There's even a **live webcam** feature for real-time detection.
>
> The stack: Python, Flask, PyTorch, OpenCV, and ReportLab.
> Real-world applications: social media moderation, journalism verification, digital forensics."

---

## 🏆 QUICK VIVA SUMMARY

- **Project**: DeepShield Advanced — AI Deepfake Detection
- **Model**: EfficientNet-B0 (Transfer Learning, ImageNet pretrained)
- **Task**: Binary Classification — Real vs Fake
- **Explainability**: GradCAM (Gradient-weighted Class Activation Mapping)
- **Backend**: Python + Flask
- **Frontend**: HTML5 + CSS3 + Vanilla JavaScript
- **Features**: Single · Batch · Webcam · Heatmap · PDF Report
- **Inference**: ~0.5 seconds on CPU

---

*Good luck at your exhibition! DeepShield is ready! 🚀*
