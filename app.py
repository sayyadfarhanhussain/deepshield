"""
DeepShield Advanced — app.py
Features: Single, Batch, Webcam, Heatmap (GradCAM), PDF Report
FIXED VERSION — All errors resolved
"""
import os, uuid, base64, traceback
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename

# ── Optional imports (won't crash if missing) ──
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("WARNING: opencv-python not installed. Heatmap feature disabled.")
    print("Fix: pip install opencv-python")

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("WARNING: reportlab not installed. PDF feature disabled.")
    print("Fix: pip install reportlab")

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']  = 'static/uploads'
app.config['REPORT_FOLDER']  = 'static/reports'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

for folder in [app.config['UPLOAD_FOLDER'],
               app.config['REPORT_FOLDER'],
               app.config['HEATMAP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        n = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n, 2)
        )

    def forward(self, x):
        return self.model(x)

# ── Lazy model loading — load on first request to save memory ──
device = torch.device('cpu')
model = None

def get_model():
    global model
    if model is not None:
        return model
    print("Loading model...")
    m = DeepfakeDetector().to(device)
    MODEL_WEIGHTS = 'deepfake_model.pth'
    if os.path.exists(MODEL_WEIGHTS):
        try:
            m.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
            print("✅ Trained weights loaded!")
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")
    else:
        print("⚠️  Using ImageNet weights")
    m.eval()
    model = m
    print("Model ready!")
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model  = model
        self.gradients  = None
        self.activations = None
        layer = model.model.features[-1]
        layer.register_forward_hook(self._save_act)
        layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, inp, cls_idx):
        self.model.zero_grad()
        out = self.model(inp)
        out[0, cls_idx].backward()
        w   = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# GradCAM initialized lazily with model

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_image(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_IMG)

def analyze(path):
    """Core detection — returns dict with scores."""
    m   = get_model()
    img = Image.open(path).convert('RGB')
    t   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(m(t), dim=1)[0]
    # Dataset mein fake=0, real=1 hai
    f = round(p[0].item() * 100, 2)   # index 0 = fake
    r = round(p[1].item() * 100, 2)   # index 1 = real
    is_fake = f > r
    return {
        'label':      'Deepfake Detected' if is_fake else 'Real Image',
        'confidence': round(f if is_fake else r, 2),
        'real_score': r,
        'fake_score': f,
        'is_fake':    is_fake
    }

def make_heatmap(path, prefix):
    """Generate GradCAM heatmap overlay. Requires opencv."""
    if not CV2_OK:
        return None, None
    m   = get_model()
    gc  = GradCAM(m)
    img = Image.open(path).convert('RGB')
    sz  = img.size
    t2  = transform(img).unsqueeze(0).to(device)
    t2.requires_grad_(True)
    with torch.no_grad():
        p = torch.softmax(m(t2), dim=1)[0]
    cls = 1 if p[1] > p[0] else 0
    t3  = transform(img).unsqueeze(0).to(device)
    t3.requires_grad_(True)
    cam = gc.generate(t3, cls)
    cam = cv2.resize(cam, sz)
    hm  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    orig    = np.array(img.resize(sz))
    overlay = cv2.addWeighted(orig, 0.55, hm, 0.45, 0)
    fn  = f"{prefix}_heatmap.jpg"
    out = os.path.join(app.config['HEATMAP_FOLDER'], fn)
    Image.fromarray(overlay).save(out)
    return out, fn

def make_pdf(results, rid):
    """Generate PDF report. Requires reportlab."""
    fn   = f"deepshield_report_{rid}.pdf"
    path = os.path.join(app.config['REPORT_FOLDER'], fn)
    doc  = SimpleDocTemplate(path, pagesize=A4,
           topMargin=1.5*cm, bottomMargin=1.5*cm,
           leftMargin=2*cm,  rightMargin=2*cm)
    styles = getSampleStyleSheet()
    s = []

    title_s = ParagraphStyle('T', fontSize=24, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#007755'),
        alignment=TA_CENTER, spaceAfter=4)
    sub_s = ParagraphStyle('S', fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER, spaceAfter=16)
    h_s = ParagraphStyle('H', fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#222222'),
        spaceBefore=12, spaceAfter=6)
    n_s = ParagraphStyle('N', fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor('#333333'),
        spaceAfter=4, leading=14)
    lbl_s = ParagraphStyle('L', fontSize=9, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#666666'))

    s.append(Spacer(1, 0.4*cm))
    s.append(Paragraph("DEEPSHIELD", title_s))
    s.append(Paragraph("AI-Powered Deepfake Detection Report", sub_s))
    s.append(HRFlowable(width="100%", thickness=2,
                        color=colors.HexColor('#00cc99')))
    s.append(Spacer(1, 0.3*cm))

    now   = datetime.now().strftime("%d %B %Y, %I:%M %p")
    total = len(results)
    fakes = sum(1 for r in results if r['result']['is_fake'])
    reals = total - fakes

    meta = [
        ['Report ID',    rid],
        ['Generated',    now],
        ['Total Images', str(total)],
        ['Real',         str(reals)],
        ['Deepfakes',    str(fakes)],
        ['Model',        'EfficientNet-B0'],
    ]
    mt = Table(meta, colWidths=[4.5*cm, 11.5*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME',      (0,0),(-1,-1), 'Helvetica'),
        ('FONTNAME',      (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,-1), 9),
        ('TEXTCOLOR',     (0,0),(0,-1),  colors.HexColor('#555')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),
            [colors.HexColor('#f8f8f8'), colors.white]),
        ('GRID',          (0,0),(-1,-1), 0.3, colors.HexColor('#ddd')),
        ('PADDING',       (0,0),(-1,-1), 7),
    ]))
    s.append(mt)
    s.append(Spacer(1, 0.5*cm))
    s.append(HRFlowable(width="100%", thickness=0.5,
                        color=colors.HexColor('#eee')))
    s.append(Paragraph("INDIVIDUAL RESULTS", h_s))

    for i, item in enumerate(results):
        r       = item['result']
        is_fake = r['is_fake']
        sc = colors.HexColor('#cc2244') if is_fake else colors.HexColor('#009966')
        bc = colors.HexColor('#fff5f7') if is_fake else colors.HexColor('#f4fff9')
        verdict = "DEEPFAKE DETECTED" if is_fake else "REAL IMAGE"

        rd = [
            [Paragraph(f"<b>#{i+1}</b>", lbl_s),
             Paragraph(f"<b>{item['filename']}</b>", lbl_s),
             Paragraph(f"<b>{verdict}</b>",
                ParagraphStyle('v', fontSize=9, fontName='Helvetica-Bold',
                               textColor=sc, alignment=TA_CENTER))],
            [Paragraph("Confidence", lbl_s),
             Paragraph(f"{r['confidence']}%", n_s), ''],
            [Paragraph("Real Score", lbl_s),
             Paragraph(f"{r['real_score']}%", n_s), ''],
            [Paragraph("Fake Score", lbl_s),
             Paragraph(f"{r['fake_score']}%", n_s), ''],
        ]
        rt = Table(rd, colWidths=[3*cm, 9*cm, 5*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',  (0,0),(-1,0),  bc),
            ('BACKGROUND',  (0,1),(-1,-1), colors.white),
            ('BOX',         (0,0),(-1,-1), 1,   colors.HexColor('#ddd')),
            ('GRID',        (0,0),(-1,-1), 0.3, colors.HexColor('#eee')),
            ('PADDING',     (0,0),(-1,-1), 7),
            ('VALIGN',      (0,0),(-1,-1), 'MIDDLE'),
        ]))
        s.append(rt)
        s.append(Spacer(1, 0.2*cm))

    s.append(Spacer(1, 0.4*cm))
    s.append(HRFlowable(width="100%", thickness=1,
                        color=colors.HexColor('#00cc99')))
    s.append(Paragraph(
        "DeepShield | Final Year Project | "
        "EfficientNet-B0 + GradCAM | PyTorch + Flask",
        ParagraphStyle('ft', fontSize=8, fontName='Helvetica',
                       textColor=colors.HexColor('#aaa'),
                       alignment=TA_CENTER)))
    doc.build(s)
    return path, fn

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


# ── SINGLE IMAGE ──
@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if not file.filename or not allowed_image(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        fn = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        file.save(fp)
        result = analyze(fp)
        return jsonify({'success': True, 'filename': fn, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── BATCH ──
@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images uploaded'}), 400
        results = []
        for file in files:
            if not file.filename:
                continue
            if not allowed_image(file.filename):
                continue
            fn = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            file.save(fp)
            result = analyze(fp)
            results.append({
                'filename':  file.filename,
                'saved_as':  fn,
                'result':    result
            })
        if not results:
            return jsonify({'error': 'No valid images found'}), 400
        return jsonify({'success': True, 'total': len(results), 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── WEBCAM ──
@app.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    try:
        data = request.get_json(force=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400

        img_data = data['image']
        # Handle both "data:image/...;base64,..." and raw base64
        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')

        fn = f"webcam_{uuid.uuid4()}.jpg"
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        img.save(fp)

        result = analyze(fp)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── HEATMAP ──
@app.route('/heatmap', methods=['POST'])
def heatmap():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if not file.filename or not allowed_image(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        uid = str(uuid.uuid4())
        fn  = uid + '_' + secure_filename(file.filename)
        fp  = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        file.save(fp)

        result = analyze(fp)
        hm_url = None

        if CV2_OK:
            try:
                _, hm_fn = make_heatmap(fp, uid)
                if hm_fn:
                    hm_url = f"/static/heatmaps/{hm_fn}"
            except Exception as he:
                print(f"Heatmap generation error: {he}")

        return jsonify({
            'success':     True,
            'filename':    fn,
            'result':      result,
            'heatmap_url': hm_url
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── PDF REPORT ──
@app.route('/report/generate', methods=['POST'])
def generate_report():
    try:
        if not REPORTLAB_OK:
            return jsonify({'error': 'reportlab not installed. Run: pip install reportlab'}), 500
        data = request.get_json(force=True)
        if not data or 'results' not in data:
            return jsonify({'error': 'No results data'}), 400
        rid = (datetime.now().strftime("%Y%m%d_%H%M%S") +
               '_' + str(uuid.uuid4())[:6].upper())
        _, rfn = make_pdf(data['results'], rid)
        return jsonify({
            'success':      True,
            'report_id':    rid,
            'download_url': f'/report/download/{rfn}'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/report/download/<filename>')
def download_report(filename):
    fp = os.path.join(app.config['REPORT_FOLDER'], secure_filename(filename))
    if not os.path.exists(fp):
        return jsonify({'error': 'File not found'}), 404
    return send_file(fp, as_attachment=True,
                     download_name=filename,
                     mimetype='application/pdf')


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DeepShield ADVANCED is RUNNING!")
    print("  Open: http://127.0.0.1:5000")
    print(f"  OpenCV  (Heatmap): {'✅ Ready' if CV2_OK else '❌ pip install opencv-python'}")
    print(f"  ReportLab (PDF)  : {'✅ Ready' if REPORTLAB_OK else '❌ pip install reportlab'}")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
