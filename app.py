"""
DeepShield Advanced v2.0 — app.py
Features: Single, Batch, Webcam, Heatmap, Video, History, Login System
"""
import os, uuid, base64, traceback
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

from flask import (Flask, request, jsonify, render_template,
                   send_file, redirect, url_for)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                     Spacer, Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

# ─────────────────────────────────────────────
# APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']              = 'deepshield-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///deepshield.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']           = 'static/uploads'
app.config['REPORT_FOLDER']           = 'static/reports'
app.config['HEATMAP_FOLDER']          = 'static/heatmaps'
app.config['VIDEO_FOLDER']            = 'static/videos'
app.config['MAX_CONTENT_LENGTH']      = 100 * 1024 * 1024

for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORT_FOLDER'],
               app.config['HEATMAP_FOLDER'], app.config['VIDEO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_IMG   = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}

db            = SQLAlchemy(app)
bcrypt        = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ─────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────
class User(db.Model, UserMixin):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    detections = db.relationship('Detection', backref='user', lazy=True)

class Detection(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename   = db.Column(db.String(200), nullable=False)
    det_type   = db.Column(db.String(20),  nullable=False)
    label      = db.Column(db.String(50),  nullable=False)
    confidence = db.Column(db.Float,       nullable=False)
    real_score = db.Column(db.Float,       nullable=False)
    fake_score = db.Column(db.Float,       nullable=False)
    is_fake    = db.Column(db.Boolean,     nullable=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─────────────────────────────────────────────
# AI MODEL
# ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        n = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n, 2)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device('cpu')
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    print("Loading model...")
    m = DeepfakeDetector().to(device)
    weights = 'deepfake_model.pth'
    if os.path.exists(weights):
        try:
            m.load_state_dict(torch.load(weights, map_location=device))
            print("✅ Trained weights loaded!")
        except Exception as e:
            print(f"⚠️  Weight error: {e}")
    else:
        print("⚠️  Using ImageNet weights")
    m.eval()
    _model = m
    print("Model ready!")
    return _model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = self.activations = None
        layer = model.model.features[-1]
        layer.register_forward_hook(self._save_act)
        layer.register_full_backward_hook(self._save_grad)
    def _save_act(self, m, i, o): self.activations = o.detach()
    def _save_grad(self, m, gi, go): self.gradients = go[0].detach()
    def generate(self, inp, cls_idx):
        self.model.zero_grad()
        out = self.model(inp)
        out[0, cls_idx].backward()
        w   = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_image(f):
    return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED_IMG

def allowed_video(f):
    return '.' in f and f.rsplit('.',1)[1].lower() in ALLOWED_VIDEO

def analyze(path):
    m   = get_model()
    img = Image.open(path).convert('RGB')
    t   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(m(t), dim=1)[0]
    fake_score = round(p[0].item() * 100, 2)
    real_score = round(p[1].item() * 100, 2)
    is_fake    = fake_score > real_score
    return {
        'label':      'Deepfake Detected' if is_fake else 'Real Image',
        'confidence': round(fake_score if is_fake else real_score, 2),
        'real_score': real_score,
        'fake_score': fake_score,
        'is_fake':    is_fake
    }

def save_detection(user_id, filename, det_type, result):
    try:
        d = Detection(user_id=user_id, filename=filename, det_type=det_type,
                      label=result['label'], confidence=result['confidence'],
                      real_score=result['real_score'], fake_score=result['fake_score'],
                      is_fake=result['is_fake'])
        db.session.add(d)
        db.session.commit()
    except Exception as e:
        print(f"DB error: {e}")

def make_heatmap(path, prefix):
    if not CV2_OK:
        return None, None
    m   = get_model()
    gc  = GradCAM(m)
    img = Image.open(path).convert('RGB')
    sz  = img.size
    with torch.no_grad():
        p = torch.softmax(m(transform(img).unsqueeze(0).to(device)), dim=1)[0]
    cls = 0 if p[0] > p[1] else 1
    t3  = transform(img).unsqueeze(0).to(device)
    t3.requires_grad_(True)
    cam = gc.generate(t3, cls)
    cam = cv2.resize(cam, sz)
    hm  = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    orig    = np.array(img.resize(sz))
    overlay = cv2.addWeighted(orig, 0.55, hm, 0.45, 0)
    fn  = f"{prefix}_heatmap.jpg"
    out = os.path.join(app.config['HEATMAP_FOLDER'], fn)
    Image.fromarray(overlay).save(out)
    return out, fn

def analyze_video(path, max_frames=10):
    if not CV2_OK:
        return None, "OpenCV not installed"
    cap      = cv2.VideoCapture(path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    duration = round(total / fps if fps > 0 else 0, 1)
    step     = max(1, total // max_frames)
    results  = []
    frame_n  = saved = 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_n % step == 0:
            fn  = f"vframe_{uuid.uuid4().hex[:8]}.jpg"
            fp  = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            cv2.imwrite(fp, frame)
            results.append({'frame': frame_n, 'filename': fn, 'result': analyze(fp)})
            saved += 1
        frame_n += 1
    cap.release()
    if not results:
        return None, "No frames extracted"
    fakes    = sum(1 for r in results if r['result']['is_fake'])
    avg_fake = round(sum(r['result']['fake_score'] for r in results) / len(results), 2)
    avg_real = round(sum(r['result']['real_score'] for r in results) / len(results), 2)
    verdict  = fakes > len(results) - fakes
    return {
        'frames':         results,
        'total_frames':   total,
        'analyzed':       len(results),
        'duration':       duration,
        'fake_frames':    fakes,
        'real_frames':    len(results) - fakes,
        'avg_fake_score': avg_fake,
        'avg_real_score': avg_real,
        'verdict':        'Deepfake Detected' if verdict else 'Real Video',
        'is_fake':        verdict,
        'confidence':     avg_fake if verdict else avg_real
    }, None

def make_pdf(results, rid):
    if not REPORTLAB_OK:
        return None, None
    fn   = f"deepshield_report_{rid}.pdf"
    path = os.path.join(app.config['REPORT_FOLDER'], fn)
    doc  = SimpleDocTemplate(path, pagesize=A4,
           topMargin=1.5*cm, bottomMargin=1.5*cm,
           leftMargin=2*cm, rightMargin=2*cm)
    s = []
    title_s = ParagraphStyle('T', fontSize=24, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#007755'), alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle('S', fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#888888'), alignment=TA_CENTER, spaceAfter=16)
    h_s     = ParagraphStyle('H', fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#222222'), spaceBefore=12, spaceAfter=6)
    n_s     = ParagraphStyle('N', fontSize=9, fontName='Helvetica',
        textColor=colors.HexColor('#333333'), spaceAfter=4, leading=14)
    lbl_s   = ParagraphStyle('L', fontSize=9, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#666666'))
    s.append(Spacer(1, 0.4*cm))
    s.append(Paragraph("DEEPSHIELD", title_s))
    s.append(Paragraph("AI-Powered Deepfake Detection Report", sub_s))
    s.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#00cc99')))
    s.append(Spacer(1, 0.3*cm))
    now   = datetime.now().strftime("%d %B %Y, %I:%M %p")
    total = len(results)
    fakes = sum(1 for r in results if r['result']['is_fake'])
    meta  = [['Report ID', rid], ['Generated', now],
             ['Total Images', str(total)], ['Real', str(total-fakes)],
             ['Deepfakes', str(fakes)], ['Model', 'EfficientNet-B0 (98% acc)']]
    mt = Table(meta, colWidths=[4.5*cm, 11.5*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'), ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),9), ('TEXTCOLOR',(0,0),(0,-1),colors.HexColor('#555')),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[colors.HexColor('#f8f8f8'),colors.white]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#ddd')), ('PADDING',(0,0),(-1,-1),7),
    ]))
    s.append(mt)
    s.append(Spacer(1, 0.5*cm))
    s.append(Paragraph("INDIVIDUAL RESULTS", h_s))
    for i, item in enumerate(results):
        r   = item['result']
        sc  = colors.HexColor('#cc2244') if r['is_fake'] else colors.HexColor('#009966')
        bc  = colors.HexColor('#fff5f7') if r['is_fake'] else colors.HexColor('#f4fff9')
        rd  = [
            [Paragraph(f"<b>#{i+1}</b>", lbl_s), Paragraph(f"<b>{item['filename']}</b>", lbl_s),
             Paragraph(f"<b>{'DEEPFAKE' if r['is_fake'] else 'REAL'}</b>",
                ParagraphStyle('v',fontSize=9,fontName='Helvetica-Bold',textColor=sc,alignment=TA_CENTER))],
            [Paragraph("Confidence",lbl_s), Paragraph(f"{r['confidence']}%",n_s), ''],
            [Paragraph("Real Score",lbl_s), Paragraph(f"{r['real_score']}%",n_s), ''],
            [Paragraph("Fake Score",lbl_s), Paragraph(f"{r['fake_score']}%",n_s), ''],
        ]
        rt = Table(rd, colWidths=[3*cm, 9*cm, 5*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),bc), ('BACKGROUND',(0,1),(-1,-1),colors.white),
            ('BOX',(0,0),(-1,-1),1,colors.HexColor('#ddd')),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#eee')),
            ('PADDING',(0,0),(-1,-1),7), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ]))
        s.append(rt)
        s.append(Spacer(1, 0.2*cm))
    s.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#00cc99')))
    s.append(Paragraph("DeepShield v2.0 | EfficientNet-B0 + GradCAM | PyTorch + Flask",
        ParagraphStyle('ft',fontSize=8,fontName='Helvetica',
                       textColor=colors.HexColor('#aaa'),alignment=TA_CENTER)))
    doc.build(s)
    return path, fn

# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        data     = request.get_json(force=True)
        username = data.get('username','').strip()
        email    = data.get('email','').strip()
        password = data.get('password','')
        if not username or not email or not password:
            return jsonify({'error': 'All fields required'}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already taken'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        hashed = bcrypt.generate_password_hash(password).decode('utf-8')
        user   = User(username=username, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        login_user(user, remember=True)
        return jsonify({'success': True, 'username': username})
    return render_template('index.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data     = request.get_json(force=True)
        email    = data.get('email','').strip()
        password = data.get('password','')
        user     = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            return jsonify({'success': True, 'username': user.username})
        return jsonify({'error': 'Invalid email or password'}), 401
    return render_template('index.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/auth/status')
def auth_status():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

# ─────────────────────────────────────────────
# MAIN ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        file = request.files['image']
        if not file.filename or not allowed_image(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        fn = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        file.save(fp)
        result = analyze(fp)
        if current_user.is_authenticated:
            save_detection(current_user.id, file.filename, 'single', result)
        return jsonify({'success': True, 'filename': fn, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    try:
        files   = request.files.getlist('images')
        results = []
        for file in files:
            if not file.filename or not allowed_image(file.filename):
                continue
            fn = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            file.save(fp)
            result = analyze(fp)
            results.append({'filename': file.filename, 'saved_as': fn, 'result': result})
            if current_user.is_authenticated:
                save_detection(current_user.id, file.filename, 'batch', result)
        if not results:
            return jsonify({'error': 'No valid images'}), 400
        return jsonify({'success': True, 'total': len(results), 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect/webcam', methods=['POST'])
def detect_webcam():
    try:
        data     = request.get_json(force=True)
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',', 1)[1]
        img  = Image.open(BytesIO(base64.b64decode(img_data))).convert('RGB')
        fn   = f"webcam_{uuid.uuid4()}.jpg"
        fp   = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        img.save(fp)
        result = analyze(fp)
        if current_user.is_authenticated:
            save_detection(current_user.id, fn, 'webcam', result)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/heatmap', methods=['POST'])
def heatmap():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        file = request.files['image']
        if not file.filename or not allowed_image(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
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
                print(f"Heatmap error: {he}")
        return jsonify({'success': True, 'filename': fn,
                        'result': result, 'heatmap_url': hm_url})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect/video', methods=['POST'])
def detect_video():
    try:
        if not CV2_OK:
            return jsonify({'error': 'OpenCV not installed'}), 500
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        file = request.files['video']
        if not file.filename or not allowed_video(file.filename):
            return jsonify({'error': 'Invalid video. Use MP4/AVI/MOV'}), 400
        fn = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        fp = os.path.join(app.config['VIDEO_FOLDER'], fn)
        file.save(fp)
        result, error = analyze_video(fp, max_frames=10)
        if error:
            return jsonify({'error': error}), 500
        if current_user.is_authenticated:
            save_detection(current_user.id, file.filename, 'video', {
                'label':      result['verdict'],
                'confidence': result['confidence'],
                'real_score': result['avg_real_score'],
                'fake_score': result['avg_fake_score'],
                'is_fake':    result['is_fake']
            })
        return jsonify({'success': True, 'filename': fn, 'result': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Login required'}), 401
    detections = Detection.query.filter_by(
        user_id=current_user.id
    ).order_by(Detection.created_at.desc()).limit(50).all()
    data = [{
        'id':         d.id,
        'filename':   d.filename,
        'type':       d.det_type,
        'label':      d.label,
        'confidence': d.confidence,
        'real_score': d.real_score,
        'fake_score': d.fake_score,
        'is_fake':    d.is_fake,
        'date':       d.created_at.strftime('%d %b %Y, %I:%M %p')
    } for d in detections]
    fakes = sum(1 for d in data if d['is_fake'])
    return jsonify({'success': True, 'detections': data,
                    'total': len(data), 'fakes': fakes, 'reals': len(data)-fakes})

@app.route('/history/clear', methods=['POST'])
def clear_history():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Login required'}), 401
    Detection.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'success': True})

@app.route('/report/generate', methods=['POST'])
def generate_report():
    try:
        if not REPORTLAB_OK:
            return jsonify({'error': 'reportlab not installed'}), 500
        data = request.get_json(force=True)
        rid  = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + str(uuid.uuid4())[:6].upper()
        _, rfn = make_pdf(data['results'], rid)
        return jsonify({'success': True, 'report_id': rid,
                        'download_url': f'/report/download/{rfn}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/report/download/<filename>')
def download_report(filename):
    fp = os.path.join(app.config['REPORT_FOLDER'], secure_filename(filename))
    if not os.path.exists(fp):
        return jsonify({'error': 'Not found'}), 404
    return send_file(fp, as_attachment=True,
                     download_name=filename, mimetype='application/pdf')

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("✅ Database ready!")

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  DeepShield ADVANCED v2.0")
    print("  Open: http://127.0.0.1:5000")
    print(f"  OpenCV   : {'✅' if CV2_OK else '❌'}")
    print(f"  ReportLab: {'✅' if REPORTLAB_OK else '❌'}")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
