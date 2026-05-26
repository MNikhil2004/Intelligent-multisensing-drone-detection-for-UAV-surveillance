"""
================================================================
DRONE DETECTION SYSTEM — DESKTOP GUI
================================================================
SETUP:
    pip install PyQt5 opencv-python torch torchvision
    pip install librosa numpy matplotlib

HOW TO RUN:
    python drone_detection_gui.py

FEATURES:
    - RGB / IR / Audio — any single or combination input
    - Approach B gated pipeline (FrameDiff for RGB, DualMOG2 for IR)
    - Individual probabilities with animated progress bars
    - Static + Dynamic fusion results
    - Annotated output video with precise bounding boxes
    - Video playback inside the GUI
    - Detection history log (Zebra-striped)
    - Cleaned Confidence gauge (0 on left, 100 on right, Times New Roman)
    - Drop Shadows & Interactive Hover UI
    - Dynamic Processing Button & Modern Scrollbars
    - Export results to CSV
================================================================
"""

import sys
import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms
import subprocess
import csv
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QTabWidget, QTextEdit, QFrame, QScrollArea,
    QGroupBox, QStatusBar, QMenu, QAction, QMessageBox,
    QSlider, QTableWidget, QTableWidgetItem, QHeaderView,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor,
    QPainter, QPen,
)

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# COLOUR PALETTE
# ================================================================
DARK_BG      = "#F5F7FA"
PANEL_BG     = "#FFFFFF"
CARD_BG      = "#F0F4F8"
BORDER       = "#CBD5E0"
ACCENT       = "#2B6CB0"
ACCENT2      = "#276749"
DRONE_RED    = "#C53030"
SAFE_GREEN   = "#276749"
TEXT_PRIMARY = "#1A202C"
TEXT_MUTED   = "#718096"
WARN_AMBER   = "#B7791F"


# ================================================================
# MODEL ARCHITECTURES
# ================================================================

class CapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_caps):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_caps, in_dim, out_dim))
    def forward(self, x):
        u   = torch.einsum("bi,kio->bko", x, self.W)
        mag = torch.norm(u, dim=-1, keepdim=True)
        return (mag**2/(1+mag**2))*(u/(mag+1e-8))

class RCapsNet_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,5,2), nn.ReLU(),
            nn.Conv2d(64,128,5,2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc   = nn.Linear(128,128)
        self.caps = CapsuleLayer(128,32,4)
    def forward(self,x):
        x=self.conv(x).view(x.size(0),-1)
        return self.caps(self.fc(x)).mean(1)

class RCapsGRU_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = RCapsNet_RGB()
        self.gru        = nn.GRU(32,32,batch_first=True)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(32,1)
    def forward(self,x):
        seq=[self.frame_encoder(x[:,t]) for t in range(x.size(1))]
        _,h=self.gru(torch.stack(seq,1))
        return self.classifier(self.dropout(h[-1])).view(-1)

class RCapsNet_IR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,5,2), nn.ReLU(),
            nn.Conv2d(64,128,5,2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc   = nn.Linear(128,128)
        self.caps = CapsuleLayer(128,32,4)
    def forward(self,x):
        x=self.conv(x).view(x.size(0),-1)
        return self.caps(self.fc(x)).mean(1)

class RCapsGRU_IR(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_encoder = RCapsNet_IR()
        self.gru        = nn.GRU(32,32,batch_first=True)
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(32,1)
    def forward(self,x):
        seq=[self.frame_encoder(x[:,t]) for t in range(x.size(1))]
        _,h=self.gru(torch.stack(seq,1))
        return self.classifier(self.dropout(h[-1])).view(-1)

class AudioCNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.gru=nn.GRU(128*16,128,batch_first=True,bidirectional=True)
        self.fc=nn.Linear(256,2)
    def forward(self,x):
        x=self.cnn(x); B,C,F,T=x.shape
        x=x.permute(0,3,1,2).reshape(B,T,C*F)
        x,_=self.gru(x)
        return self.fc(x[:,-1])


# ================================================================
# INFERENCE ENGINE — Approach B
# ================================================================

rgb_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
ir_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

def build_framediff_mask(gray_curr, gray_prev, clahe):
    diff   = cv2.absdiff(clahe.apply(gray_curr), clahe.apply(gray_prev))
    _,mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    k1     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    k2     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return cv2.dilate(mask, k2, iterations=1)

def build_dualmog2_mask(gray, bg_slow, bg_fast, clahe):
    enhanced = clahe.apply(gray)
    m1=bg_slow.apply(enhanced); m2=bg_slow.apply(gray)
    m3=bg_fast.apply(enhanced); m4=bg_fast.apply(gray)
    mask = cv2.bitwise_or(cv2.bitwise_or(m1,m2), cv2.bitwise_or(m3,m4))
    k1   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    k2   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return cv2.dilate(mask, k2, iterations=1)

def build_kalman():
    kf=cv2.KalmanFilter(4,2)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kf.processNoiseCov     = np.eye(4,dtype=np.float32)*0.03
    kf.measurementNoiseCov = np.eye(2,dtype=np.float32)*1.0
    kf.errorCovPost        = np.eye(4,dtype=np.float32)
    return kf

def get_valid_contours(mask, min_area=8, max_area=15000):
    c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return [x for x in c if min_area<cv2.contourArea(x)<max_area]

def get_best_contour(contours, ki, kf):
    if not contours: return None
    if ki:
        p=kf.predict()
        px=float(p[0,0]); py=float(p[1,0])
        return min(contours,key=lambda c:(
            abs(cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]//2-px)+
            abs(cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3]//2-py)))
    return max(contours,key=cv2.contourArea)

def get_motion_indices_framediff(video_path):
    cap=cv2.VideoCapture(video_path)
    clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(4,4))
    indices=[]; prev=None; fid=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mask=build_framediff_mask(gray,prev,clahe)
            c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if any(8<cv2.contourArea(x)<15000 for x in c): indices.append(fid)
        prev=gray.copy(); fid+=1
    cap.release(); return indices

def get_motion_indices_dualmog2(video_path):
    cap=cv2.VideoCapture(video_path)
    clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(4,4))
    bg_slow=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=12,detectShadows=False)
    bg_fast=cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=10,detectShadows=False)
    for _ in range(15):
        ret,f=cap.read()
        if not ret: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); bg_slow.apply(g); bg_fast.apply(g)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    indices=[]; fid=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask=build_dualmog2_mask(gray,bg_slow,bg_fast,clahe)
        c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if any(8<cv2.contourArea(x)<15000 for x in c): indices.append(fid)
        fid+=1
    cap.release(); return indices

def read_frames_at_indices(video_path, indices, modality="RGB"):
    if not indices: return []
    cap=cv2.VideoCapture(video_path); idx_set=set(indices); frames=[]; fid=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        if fid in idx_set:
            if modality=="RGB":
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=cv2.resize(frame,(128,128))
            else:
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame=cv2.resize(frame,(128,128))
            frames.append(frame)
        fid+=1
    cap.release(); return frames

def compute_prob_rgb(video_path, model):
    indices=get_motion_indices_framediff(video_path)
    if len(indices)<5: return None, 0
    sampled=[indices[i] for i in np.linspace(0,len(indices)-1,40,dtype=int)]
    frames=read_frames_at_indices(video_path,sampled,"RGB")
    if not frames: return None, 0
    while len(frames)<40: frames.append(frames[-1])
    tensor=torch.stack([rgb_tf(f) for f in frames[:40]]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p=float(torch.sigmoid(model(tensor)).item())
    return p, len(indices)

def compute_prob_ir(video_path, model):
    indices=get_motion_indices_dualmog2(video_path)
    if len(indices)<5: return None, 0
    sampled=[indices[i] for i in np.linspace(0,len(indices)-1,40,dtype=int)]
    frames=read_frames_at_indices(video_path,sampled,"IR")
    if not frames: return None, 0
    while len(frames)<40: frames.append(frames[-1])
    tensor=torch.stack([ir_tf(f[:,:,None]) for f in frames[:40]]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p=float(torch.sigmoid(model(tensor)).item())
    return p, len(indices)

def compute_prob_audio(audio_path, model):
    if not LIBROSA_OK: return None, [], None, 16000, None
    y,sr=librosa.load(audio_path,sr=16000)
    y,_=librosa.effects.trim(y,top_db=40)
    SEG=4*16000; HOP=2*16000; probs=[]; last_mel=None
    for i in range(0,len(y)-SEG+1,HOP):
        mel=librosa.feature.melspectrogram(y=y[i:i+SEG],sr=16000,n_mels=128,n_fft=1024,hop_length=256)
        logmel=librosa.power_to_db(mel)[:,:247]; last_mel=logmel
        x=torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            probs.append(torch.softmax(model(x),1)[0,1].item())
    avg = float(np.mean(probs)) if probs else None
    return avg, probs, y, sr, last_mel

def fuse_static(pa, pr, pi):
    wa,wr,wi=0.25,0.40,0.35; total=0.0; ws=0.0
    if pa is not None: total+=pa*wa; ws+=wa
    if pr is not None: total+=pr*wr; ws+=wr
    if pi is not None: total+=pi*wi; ws+=wi
    return total/ws if ws>0 else None

def fuse_dynamic(pa, pr, pi):
    def c(p): return abs(p-0.5)*2
    total=0.0; ws=0.0
    if pa is not None: w=c(pa); total+=pa*w; ws+=w
    if pr is not None: w=c(pr); total+=pr*w; ws+=w
    if pi is not None: w=c(pi); total+=pi*w; ws+=w
    return total/ws if ws>0 else None

def generate_annotated_video(video_path, prob, modality, output_path, progress_cb=None):
    is_drone  = prob >= 0.5
    label     = "DRONE"    if is_drone else "NOT DRONE"
    box_color = (0,50,255) if is_drone else (0,200,80)
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clahe=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(4,4))
    kf=build_kalman(); ki=False; prev=None
    if modality=="IR":
        bg_slow=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=12,detectShadows=False)
        bg_fast=cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=10,detectShadows=False)
        for _ in range(15):
            ret,f=cap.read()
            if not ret: break
            g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); bg_slow.apply(g); bg_fast.apply(g)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    tmp=output_path.replace(".mp4","_tmp.mp4")
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out=cv2.VideoWriter(tmp,fourcc,fps,(W,H))
    fid=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        frame_bgr=(cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR) if modality=="IR" and len(frame.shape)==2 else frame.copy())
        gray=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2GRAY)
        if modality=="RGB":
            if prev is None: prev=gray.copy(); out.write(frame_bgr); fid+=1; continue
            mask=build_framediff_mask(gray,prev,clahe)
        else:
            mask=build_dualmog2_mask(gray,bg_slow,bg_fast,clahe)
        valid=get_valid_contours(mask,min_area=8 if modality=="RGB" else 20)
        best=get_best_contour(valid,ki,kf)
        meas=None
        if best is not None:
            bx,by,bw,bh=cv2.boundingRect(best)
            cx,cy=bx+bw//2,by+bh//2
            meas=np.array([[np.float32(cx)],[np.float32(cy)]])
        if meas is not None:
            if not ki:
                kf.statePre=np.array([[meas[0,0]],[meas[1,0]],[0.],[0.]],np.float32); ki=True
            kf.correct(meas)
        if ki: kf.predict()
        if valid:
            bst=max(valid,key=cv2.contourArea)
            bx,by,bw,bh=cv2.boundingRect(bst)
            bx=max(bx,0); by=max(by,0); bw=min(bw,W-bx); bh=min(bh,H-by)
            cv2.rectangle(frame_bgr,(bx,by),(bx+bw,by+bh),box_color,2)
            L=20
            for (sx,sy,dx,dy) in [(bx,by,1,1),(bx+bw,by,-1,1),(bx,by+bh,1,-1),(bx+bw,by+bh,-1,-1)]:
                cv2.line(frame_bgr,(sx,sy),(sx+dx*L,sy),box_color,3)
                cv2.line(frame_bgr,(sx,sy),(sx,sy+dy*L),box_color,3)
            tag=f"{label}  {prob:.2f}"
            (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
            cv2.rectangle(frame_bgr,(bx,by-th-10),(bx+tw+6,by),box_color,-1)
            cv2.putText(frame_bgr,tag,(bx+3,by-4),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
        bar_w=int(W*prob)
        cv2.rectangle(frame_bgr,(0,H-8),(W,H),(20,20,30),-1)
        cv2.rectangle(frame_bgr,(0,H-8),(bar_w,H),box_color,-1)
        out.write(frame_bgr)
        if modality=="RGB": prev=gray.copy()
        fid+=1
        if progress_cb and total>0: progress_cb(int(fid/total*100))
    cap.release(); out.release()
    try:
        subprocess.run(["ffmpeg","-y","-i",tmp,"-vcodec","libx264","-crf","23",
                        "-preset","fast",output_path,"-loglevel","quiet"],check=True)
        os.remove(tmp)
    except Exception:
        if os.path.exists(tmp): 
            os.replace(tmp, output_path)
    return output_path


# ================================================================
# AUDIO VISUALIZER
# ================================================================

def generate_audio_visualization(audio_path, model, device):
    if not MATPLOTLIB_OK or not LIBROSA_OK: return None
    y, sr = librosa.load(audio_path, sr=16000)
    y, _  = librosa.effects.trim(y, top_db=40)
    SEG = 4*16000; HOP = 2*16000
    seg_probs=[]; seg_times=[]
    for i in range(0, len(y)-SEG+1, HOP):
        mel    = librosa.feature.melspectrogram(y=y[i:i+SEG],sr=sr,n_mels=128,n_fft=1024,hop_length=256)
        logmel = librosa.power_to_db(mel)[:,:247]
        x      = torch.tensor(logmel).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            p = torch.softmax(model(x),1)[0,1].item()
        seg_probs.append(p); seg_times.append(i/sr)
    if not seg_probs: return None
    mel_full = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,n_fft=1024,hop_length=256)
    mel_db   = librosa.power_to_db(mel_full, ref=np.max)
    duration = len(y)/sr
    times_wave = np.linspace(0, duration, len(y))
    fig = plt.figure(figsize=(12,8), facecolor="white")
    fig.suptitle(f"Audio Analysis — {os.path.basename(audio_path)}",
                 fontsize=13, fontweight="bold", color="#1E293B", y=0.98)
    gs = gridspec.GridSpec(3,1,figure=fig,hspace=0.55,top=0.93,bottom=0.07,left=0.08,right=0.97)
    ax1=fig.add_subplot(gs[0])
    ax1.plot(times_wave,y,color="#1D4ED8",linewidth=0.6,alpha=0.85)
    ax1.axhline(0,color="#CBD5E1",linewidth=0.5,linestyle="--")
    ax1.fill_between(times_wave,y,0,where=(y>0),color="#BFDBFE",alpha=0.4)
    ax1.fill_between(times_wave,y,0,where=(y<0),color="#FEE2E2",alpha=0.4)
    ax1.set_title("Waveform (Amplitude vs Time)",fontsize=10,fontweight="bold",color="#1E293B",pad=6)
    ax1.set_xlabel("Time (s)",fontsize=8,color="#64748B"); ax1.set_ylabel("Amplitude",fontsize=8,color="#64748B")
    ax1.set_xlim(0,duration); ax1.tick_params(labelsize=7,colors="#64748B")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.set_facecolor("#FAFAFA")
    for sp in ["left","bottom"]: ax1.spines[sp].set_color("#CBD5E1")
    ax2=fig.add_subplot(gs[1])
    img=ax2.imshow(mel_db,aspect="auto",origin="lower",extent=[0,duration,0,sr/2/1000],cmap="Blues",interpolation="nearest")
    cbar=fig.colorbar(img,ax=ax2,pad=0.01,shrink=0.95)
    cbar.ax.tick_params(labelsize=7); cbar.set_label("dB",fontsize=7,color="#64748B")
    ax2.set_title("Mel Spectrogram (Model Input)",fontsize=10,fontweight="bold",color="#1E293B",pad=6)
    ax2.set_xlabel("Time (s)",fontsize=8,color="#64748B"); ax2.set_ylabel("Freq (kHz)",fontsize=8,color="#64748B")
    ax2.tick_params(labelsize=7,colors="#64748B"); ax2.set_facecolor("#EFF6FF")
    for sp in ax2.spines.values(): sp.set_color("#CBD5E1")
    ax3=fig.add_subplot(gs[2])
    seg_centers=[t+(SEG/sr)/2 for t in seg_times]
    bar_colors=[DRONE_RED if p>=0.5 else SAFE_GREEN for p in seg_probs]
    bars=ax3.bar(seg_centers,seg_probs,width=(HOP/sr)*0.8,color=bar_colors,alpha=0.85,edgecolor="white",linewidth=0.8)
    ax3.axhline(0.5,color="#D97706",linewidth=1.5,linestyle="--",zorder=5)
    for bar,pp in zip(bars,seg_probs):
        ax3.text(bar.get_x()+bar.get_width()/2,min(pp+0.03,0.95),f"{pp:.2f}",
                 ha="center",va="bottom",fontsize=7,color="#1E293B",fontweight="bold")
    ax3.set_ylim(0,1.05); ax3.set_xlim(0,duration)
    ax3.set_title("Probability per Segment (4s window, 2s hop)",fontsize=10,fontweight="bold",color="#1E293B",pad=6)
    ax3.set_xlabel("Time (s)",fontsize=8,color="#64748B"); ax3.set_ylabel("Drone P",fontsize=8,color="#64748B")
    ax3.tick_params(labelsize=7,colors="#64748B"); ax3.set_facecolor("#FAFAFA")
    for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax3.spines[sp].set_color("#CBD5E1")
    from matplotlib.patches import Patch
    ax3.legend(handles=[
        Patch(color=DRONE_RED,alpha=0.85,label="DRONE detected"),
        Patch(color=SAFE_GREEN,alpha=0.85,label="Not drone"),
        plt.Line2D([0],[0],color="#D97706",linestyle="--",linewidth=1.5,label="Threshold (0.5)"),
    ],fontsize=8,loc="upper right",framealpha=0.9,edgecolor="#CBD5E1")
    canvas=FigureCanvasAgg(fig); canvas.draw()
    buf=canvas.buffer_rgba(); w,h=canvas.get_width_height()
    img_array=np.frombuffer(buf,dtype=np.uint8).reshape(h,w,4).copy()
    img_rgb=np.ascontiguousarray(img_array[:,:,:3])
    qimg=QImage(img_rgb.tobytes(),w,h,w*3,QImage.Format_RGB888)
    pixmap=QPixmap.fromImage(qimg); plt.close(fig)
    return pixmap


# ================================================================
# INFERENCE WORKER THREAD
# ================================================================

class InferenceWorker(QThread):
    progress    = pyqtSignal(int, str)
    result      = pyqtSignal(dict)
    error       = pyqtSignal(str)
    video_ready = pyqtSignal(str, str)

    def __init__(self, rgb_path, ir_path, audio_path,
                 rgb_model, ir_model, audio_model, output_dir):
        super().__init__()
        self.rgb_path=rgb_path; self.ir_path=ir_path; self.audio_path=audio_path
        self.rgb_model=rgb_model; self.ir_model=ir_model; self.audio_model=audio_model
        self.output_dir=output_dir

    def run(self):
        try:
            results={}
            if self.audio_path and self.audio_model:
                self.progress.emit(5,"Processing audio...")
                p,seg_probs,waveform,sr,mel_db=compute_prob_audio(self.audio_path,self.audio_model)
                results.update({'audio':p,'audio_segs':seg_probs,'audio_wave':waveform,'audio_sr':sr,'audio_mel':mel_db})
                self.progress.emit(18,"Generating audio visualization...")
                results['audio_pixmap']=generate_audio_visualization(self.audio_path,self.audio_model,DEVICE)
                self.progress.emit(22,f"Audio probability: {p:.4f}")
            else:
                results['audio']=None
            if self.rgb_path and self.rgb_model:
                self.progress.emit(25,"FrameDiff scanning RGB for motion frames...")
                p,n_motion=compute_prob_rgb(self.rgb_path,self.rgb_model)
                results['rgb']=p; results['rgb_motion_frames']=n_motion
                self.progress.emit(50,f"RGB probability: {p:.4f}  (motion frames: {n_motion})")
                self.progress.emit(52,"Generating RGB annotated video...")
                rgb_stem=os.path.splitext(os.path.basename(self.rgb_path))[0]
                out_rgb=os.path.join(self.output_dir,f"output_{rgb_stem}.mp4")
                generate_annotated_video(self.rgb_path,p,"RGB",out_rgb,
                    lambda x: self.progress.emit(52+int(x*0.18),"Rendering RGB video..."))
                self.video_ready.emit("RGB",out_rgb); self.progress.emit(70,"RGB video ready")
            else:
                results['rgb']=None; results['rgb_motion_frames']=0
            if self.ir_path and self.ir_model:
                self.progress.emit(72,"DualMOG2 scanning IR for motion frames...")
                p,n_motion=compute_prob_ir(self.ir_path,self.ir_model)
                results['ir']=p; results['ir_motion_frames']=n_motion
                self.progress.emit(85,f"IR probability: {p:.4f}  (motion frames: {n_motion})")
                self.progress.emit(87,"Generating IR annotated video...")
                ir_stem=os.path.splitext(os.path.basename(self.ir_path))[0]
                out_ir=os.path.join(self.output_dir,f"output_{ir_stem}.mp4")
                generate_annotated_video(self.ir_path,p,"IR",out_ir,
                    lambda x: self.progress.emit(87+int(x*0.1),"Rendering IR video..."))
                self.video_ready.emit("IR",out_ir); self.progress.emit(97,"IR video ready")
            else:
                results['ir']=None; results['ir_motion_frames']=0
            results['static_fusion'] =fuse_static(results['audio'],results['rgb'],results['ir'])
            results['dynamic_fusion']=fuse_dynamic(results['audio'],results['rgb'],results['ir'])
            results['timestamp']     =datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.progress.emit(100,"Detection complete!")
            self.result.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ================================================================
# CUSTOM WIDGETS
# ================================================================

class ProbabilityBar(QWidget):
    """Animated probability bar row — larger fonts"""
    def __init__(self, name, subtitle, color, parent=None):
        super().__init__(parent)
        self.name=name; self.subtitle=subtitle; self.color=color
        self.value=0.0; self.target=0.0; self.active=False
        self.setMinimumHeight(82)
        self.timer=QTimer(); self.timer.timeout.connect(self._animate)

    def setValue(self, v):
        self.target=max(0.0,min(1.0,v)); self.active=True; self.timer.start(16)

    def reset(self):
        self.value=0.0; self.target=0.0; self.active=False; self.update()

    def _animate(self):
        diff=self.target-self.value
        if abs(diff)<0.002: self.value=self.target; self.timer.stop()
        else: self.value+=diff*0.10
        self.update()

    def paintEvent(self, event):
        painter=QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W=self.width(); H=self.height()
        painter.fillRect(0,0,W,H,QColor(PANEL_BG))
        painter.setPen(QPen(QColor(BORDER),1))
        painter.drawLine(0,H-1,W,H-1)

        LEFT=20; RIGHT=20; NAME_W=360; VAL_W=120
        BAR_X=LEFT+NAME_W+16; BAR_W=W-BAR_X-VAL_W-RIGHT
        BAR_Y=H//2-6; BAR_H=12

        # Name
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.setFont(QFont("Times New Roman",14,QFont.Bold))
        painter.drawText(LEFT, H//2-6, self.name)

        # Subtitle
        painter.setPen(QColor(TEXT_MUTED))
        painter.setFont(QFont("Consolas",11))
        painter.drawText(LEFT, H//2+16, self.subtitle)

        # Track
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(DARK_BG))
        painter.drawRoundedRect(BAR_X,BAR_Y,BAR_W,BAR_H,6,6)

        # Fill
        if self.active and self.value>0:
            fill_w=max(8,int(BAR_W*self.value))
            painter.setBrush(QColor(self.color))
            painter.drawRoundedRect(BAR_X,BAR_Y,fill_w,BAR_H,6,6)

        # Threshold tick
        tick_x=BAR_X+BAR_W//2
        painter.setPen(QPen(QColor("#A0AEC0"),1))
        painter.drawLine(tick_x,BAR_Y-6,tick_x,BAR_Y+BAR_H+6)

        # Value
        if self.active:
            pct=self.value*100
            is_drone=self.value>=0.5
            val_color=QColor(DRONE_RED) if is_drone else QColor(SAFE_GREEN)
            painter.setPen(val_color)
            painter.setFont(QFont("Consolas",16,QFont.Bold))
            painter.drawText(W-VAL_W-RIGHT, H//2-4, f"{pct:.1f}%")
            painter.setFont(QFont("Consolas",11))
            painter.drawText(W-VAL_W-RIGHT, H//2+16, "drone" if is_drone else "not drone")
        else:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("Consolas",15))
            painter.drawText(W-VAL_W-RIGHT, H//2+6, "—")
        painter.end()


class VideoPlayer(QWidget):
    """Embedded video player"""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title=title; self.cap=None
        self.timer=QTimer(); self.timer.timeout.connect(self._next_frame)
        self._setup_ui()

    def _setup_ui(self):
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(4)
        title_lbl=QLabel(self.title)
        title_lbl.setStyleSheet(f"color:{ACCENT}; font:bold 12px 'Consolas'; padding:4px;")
        layout.addWidget(title_lbl)
        self.frame_lbl=QLabel()
        self.frame_lbl.setMinimumSize(380,240)
        self.frame_lbl.setAlignment(Qt.AlignCenter)
        self.frame_lbl.setStyleSheet(
            f"background:{DARK_BG}; border:1px solid {BORDER}; border-radius:6px; color:{TEXT_MUTED};")
        self.frame_lbl.setText("No video loaded")
        layout.addWidget(self.frame_lbl)
        ctrl=QHBoxLayout()
        self.btn_play =self._btn("▶ Play", self._play)
        self.btn_pause=self._btn("⏸ Pause",self._pause)
        self.btn_stop =self._btn("⏹ Stop", self._stop)
        ctrl.addWidget(self.btn_play); ctrl.addWidget(self.btn_pause); ctrl.addWidget(self.btn_stop)
        layout.addLayout(ctrl)
        self.slider=QSlider(Qt.Horizontal)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{DARK_BG}; height:4px; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; height:14px; border-radius:7px; margin:-5px 0; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}""")
        layout.addWidget(self.slider)

    def _btn(self, text, cb):
        b=QPushButton(text); b.clicked.connect(cb)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{ background:{PANEL_BG}; color:{TEXT_PRIMARY};
                border:1px solid {BORDER}; border-radius:4px; padding:5px 12px; font:11px 'Consolas'; }}
            QPushButton:hover {{ background:{CARD_BG}; border-color:{ACCENT}; }}""")
        return b

    def load(self, path):
        if self.cap: self.cap.release()
        self.cap=cv2.VideoCapture(path)
        self.total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps=self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.slider.setMaximum(max(1,self.total-1)); self.slider.setValue(0)
        self._show_frame()

    def _show_frame(self):
        if not self.cap: return
        ret,frame=self.cap.read()
        if not ret: self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); self._pause(); return
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c=frame.shape
        img=QImage(frame.data,w,h,c*w,QImage.Format_RGB888)
        pix=QPixmap.fromImage(img).scaled(
            self.frame_lbl.width(),self.frame_lbl.height(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        self.frame_lbl.setPixmap(pix)

    def _next_frame(self): self._show_frame()
    def _play(self):
        if self.cap: self.timer.start(int(1000/max(1,self.fps)))
    def _pause(self): self.timer.stop()
    def _stop(self):
        self.timer.stop()
        if self.cap: self.cap.set(cv2.CAP_PROP_POS_FRAMES,0); self._show_frame()


# ================================================================
# CONFIDENCE GAUGE  — LEFT → RIGHT (TIMES NEW ROMAN)
# ================================================================

class ConfidenceGauge(QWidget):
    """
    Semicircle gauge using Times New Roman for a classic, clean aesthetic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value  = 0.0
        self.target = 0.0
        self.setMinimumSize(480, 230)
        self.setMaximumHeight(260)
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)

    def setValue(self, v):
        self.target = max(0.0, min(1.0, v))
        self.timer.start(16)

    def reset(self):
        self.value = 0.0; self.target = 0.0; self.update()

    def _animate(self):
        diff = self.target - self.value
        if abs(diff) < 0.002:
            self.value = self.target; self.timer.stop()
        else:
            self.value += diff * 0.10
        self.update()

    def paintEvent(self, event):
        import math
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W = self.width(); H = self.height()
        painter.fillRect(0, 0, W, H, QColor(PANEL_BG))

        cx = W // 2
        cy = H - 30 
        
        r = min(W // 2 - 50, cy - 35)
        r = max(r, 80)
        THICK = 18  

        rect = QRect(cx - r, cy - r, 2*r, 2*r)

        # ── Background track ──
        pen_track = QPen(QColor("#EDF2F7"), THICK)
        pen_track.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(rect, 0 * 16, 180 * 16)

        # ── Filled arc ──
        is_drone  = self.value >= 0.5
        arc_color = QColor(DRONE_RED) if is_drone else QColor(SAFE_GREEN)

        if self.value > 0:
            pen_fill = QPen(arc_color, THICK)
            pen_fill.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_fill)
            span = -int(self.value * 180 * 16)
            painter.drawArc(rect, 180 * 16, span)

        # ── Labels ──
        painter.setFont(QFont("Times New Roman", 14, QFont.Bold))
        painter.setPen(QColor("#4A5568")) 

        for pct in [0, 25, 50, 75, 100]:
            qt_angle_deg  = 180 - pct * 1.8        
            math_angle_rad = math.radians(qt_angle_deg)
            cos_a = math.cos(math_angle_rad)
            sin_a = math.sin(math_angle_rad)

            label_r = r + 24
            lx = cx + label_r * cos_a
            ly = cy - label_r * sin_a

            rect_w, rect_h = 40, 24
            text_rect = QRect(int(lx - rect_w/2), int(ly - rect_h/2), rect_w, rect_h)
            painter.drawText(text_rect, Qt.AlignCenter, str(pct))

        # ── Large percentage text ──
        val_color = arc_color if self.value > 0 else QColor(TEXT_MUTED)
        painter.setPen(val_color)
        painter.setFont(QFont("Times New Roman", 44, QFont.Bold))
        painter.drawText(QRect(cx - 100, cy - 70, 200, 60), Qt.AlignCenter,
                         f"{self.value * 100:.1f}%")

        # ── "confidence" label ──
        painter.setPen(QColor(TEXT_MUTED))
        painter.setFont(QFont("Times New Roman", 14, QFont.StyleItalic))
        painter.drawText(QRect(cx - 80, cy - 10, 160, 20), Qt.AlignCenter, "confidence")

        painter.end()


# ================================================================
# VERDICT CARD
# ================================================================

class VerdictCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._set_neutral_style()
        
        # Interactive glowing shadow
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setColor(QColor(0, 0, 0, 30))
        self.shadow.setOffset(0, 5)
        self.setGraphicsEffect(self.shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(6)

        lbl_title = QLabel("FINAL ASSESSMENT")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet(
            f"color:{TEXT_MUTED}; font:bold 12px 'Consolas'; letter-spacing:2px;"
            f"border:none; background:transparent;")
        layout.addWidget(lbl_title)

        self.lbl_verdict = QLabel("Awaiting input")
        self.lbl_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_verdict.setStyleSheet(
            f"color:{TEXT_MUTED}; font:bold 28px 'Times New Roman'; border:none; background:transparent;")
        layout.addWidget(self.lbl_verdict)

        self.lbl_conf = QLabel("")
        self.lbl_conf.setAlignment(Qt.AlignCenter)
        self.lbl_conf.setStyleSheet(
            f"color:{TEXT_MUTED}; font:14px 'Consolas'; border:none; background:transparent;")
        layout.addWidget(self.lbl_conf)

        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet(
            f"color:{TEXT_MUTED}; font:12px 'Consolas'; border:none; background:transparent;")
        layout.addWidget(self.lbl_info)

    def _set_neutral_style(self):
        self.setStyleSheet(
            f"background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;")

    def update_result(self, verdict, conf_pct, motion_info, is_drone):
        color      = DRONE_RED if is_drone else SAFE_GREEN
        card_bg    = "#FFF5F5" if is_drone else "#F0FFF4"
        card_bdr   = DRONE_RED if is_drone else SAFE_GREEN
        self.setStyleSheet(
            f"background:{card_bg}; border:2px solid {card_bdr}; border-radius:10px;")
        self.lbl_verdict.setText(verdict)
        self.lbl_verdict.setStyleSheet(
            f"color:{color}; font:bold 28px 'Times New Roman'; border:none; background:transparent;")
        self.lbl_conf.setText(f"{conf_pct:.1f}% confidence")
        self.lbl_conf.setStyleSheet(
            f"color:{color}; font:14px 'Consolas'; border:none; background:transparent;")
        self.lbl_info.setText(motion_info)

        # Pulse Glow based on result
        if is_drone:
            self.shadow.setColor(QColor(197, 48, 48, 120))
            self.shadow.setBlurRadius(35)
        else:
            self.shadow.setColor(QColor(39, 103, 73, 120))
            self.shadow.setBlurRadius(35)

    def reset(self):
        self._set_neutral_style()
        self.lbl_verdict.setText("Awaiting input")
        self.lbl_verdict.setStyleSheet(
            f"color:{TEXT_MUTED}; font:bold 28px 'Times New Roman'; border:none; background:transparent;")
        self.lbl_conf.setText("")
        self.lbl_info.setText("")
        self.shadow.setColor(QColor(0, 0, 0, 30))
        self.shadow.setBlurRadius(20)


# ================================================================
# MAIN WINDOW
# ================================================================

class DroneDetectionGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.rgb_path=""; self.ir_path=""; self.audio_path=""
        self.rgb_model=None; self.ir_model=None; self.audio_model=None
        self.worker=None; self.results={}; self.history=[]
        self.output_dir=os.path.join(os.path.expanduser("~"),"DroneDetectionOutputs")
        os.makedirs(self.output_dir,exist_ok=True)

        self.RGB_MODEL_PATH   = r'N:\major-project\GUI\rcaps_gru_best (1).pth'
        self.IR_MODEL_PATH    = r'N:\major-project\GUI\rcaps_gru_ir_best (1).pth'
        self.AUDIO_MODEL_PATH = r'N:\major-project\GUI\audio_cnn_gru_binary_weighted (1) (1).pth'

        self._apply_theme()
        self._setup_ui()
        self._setup_menu()
        self.setWindowTitle("Drone Detection System — Approach B Gated Pipeline")
        self.setMinimumSize(1440, 860)
        self.showMaximized()

    def _apply_shadow(self, widget, blur_radius=20, alpha=35, offset=(0, 4)):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QColor(0, 0, 0, alpha))
        shadow.setOffset(offset[0], offset[1])
        widget.setGraphicsEffect(shadow)

    # ──────────────────────────────────────────────────────────
    # UI SETUP
    # ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        central=QWidget(); self.setCentralWidget(central)
        root=QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        root.addWidget(self._build_header())
        body=QHBoxLayout(); body.setContentsMargins(10,10,10,10); body.setSpacing(10)
        
        left_panel = self._build_left_panel()
        self._apply_shadow(left_panel, blur_radius=25, alpha=40, offset=(3, 0))
        body.addWidget(left_panel, stretch=2)

        right_panel = self._build_right_panel()
        self._apply_shadow(right_panel, blur_radius=20, alpha=25, offset=(0, 4))
        body.addWidget(right_panel, stretch=5)

        root.addLayout(body)
        self.status_bar=QStatusBar()
        self.status_bar.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_MUTED}; font:12px 'Consolas';")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            f"Ready  |  Device: {DEVICE}  |  Pipeline: Approach B (FrameDiff→RGB, DualMOG2→IR)")

    # ── Header ────────────────────────────────────────────────

    def _build_header(self):
        header=QFrame(); header.setFixedHeight(60)
        header.setStyleSheet(f"background:{PANEL_BG}; border-bottom:2px solid {BORDER};")
        self._apply_shadow(header, blur_radius=15, alpha=20, offset=(0, 2))
        
        lay=QHBoxLayout(header); lay.setContentsMargins(28,0,28,0)
        logo=QLabel("Drone Detection System")
        logo.setStyleSheet(f"color:{TEXT_PRIMARY}; font:bold 22px 'Times New Roman';")
        lay.addWidget(logo)
        pipe=QLabel("  APPROACH B  ·  GATED INFERENCE PIPELINE")
        pipe.setStyleSheet(f"color:{TEXT_MUTED}; font:12px 'Consolas'; letter-spacing:1px;")
        lay.addWidget(pipe); lay.addStretch()
        self.lbl_header_models=QLabel("")
        self.lbl_header_models.setStyleSheet(f"color:{SAFE_GREEN}; font:12px 'Consolas';")
        lay.addWidget(self.lbl_header_models); lay.addSpacing(20)
        dev=QLabel(f"  {str(DEVICE).upper()}  ")
        dev.setStyleSheet(
            f"color:{TEXT_PRIMARY}; background:{DARK_BG}; border:1px solid {BORDER};"
            f"border-radius:4px; font:bold 12px 'Consolas'; padding:5px 10px;")
        lay.addWidget(dev)
        return header

    # ── Left Panel — wider ─────────────────────────────────────

    def _build_left_panel(self):
        panel=QFrame()
        panel.setMinimumWidth(440)
        panel.setMaximumWidth(550)
        panel.setStyleSheet(f"background:{PANEL_BG}; border-right:1px solid {BORDER}; border-radius: 8px;")
        lay=QVBoxLayout(panel); lay.setContentsMargins(18,18,18,18); lay.setSpacing(16)

        # Model Status
        grp_model=self._group("MODEL STATUS")
        ml=QVBoxLayout(grp_model); ml.setSpacing(10)
        self.btn_load_models=self._action_btn("Load Models", self._load_models)
        self.lbl_model_status=QLabel(
            "RCapsGRU_RGB   not loaded\nRCapsGRU_IR    not loaded\nAudioCNNGRU    not loaded")
        self.lbl_model_status.setStyleSheet(f"color:{TEXT_MUTED}; font:13px 'Consolas'; line-height:1.8;")
        ml.addWidget(self.btn_load_models); ml.addWidget(self.lbl_model_status)
        lay.addWidget(grp_model)

        # Input Files
        grp_input=self._group("INPUT FILES")
        il=QVBoxLayout(grp_input); il.setSpacing(5)
        self.btn_rgb  =self._file_btn("📷  Select RGB Video (.mp4)", self._select_rgb)
        self.lbl_rgb  =self._path_label("No file selected")
        self.btn_ir   =self._file_btn("🌡  Select IR Video  (.mp4)", self._select_ir)
        self.lbl_ir   =self._path_label("No file selected")
        self.btn_audio=self._file_btn("🎙  Select Audio     (.wav)", self._select_audio)
        self.lbl_audio=self._path_label("No file selected")
        for w in [self.btn_rgb,self.lbl_rgb,self.btn_ir,self.lbl_ir,self.btn_audio,self.lbl_audio]:
            il.addWidget(w)
        lay.addWidget(grp_input)

        # Dynamic Run Button
        self.btn_run=QPushButton("▶   RUN DETECTION")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.setMinimumHeight(54)
        self.btn_run.clicked.connect(self._run_detection)
        self._reset_run_button() # Applies the initial styling
        lay.addWidget(self.btn_run)

        # Progress
        self.progress=QProgressBar()
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background:{DARK_BG}; border:1px solid {BORDER}; border-radius:5px;
                height:22px; text-align:center; color:{TEXT_PRIMARY}; font:13px 'Consolas';
            }}
            QProgressBar::chunk {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {ACCENT2},stop:1 {ACCENT});
                border-radius:4px;
            }}""")
        self.progress.setValue(0); lay.addWidget(self.progress)

        self.lbl_status=QLabel("Awaiting input...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color:{TEXT_MUTED}; font:13px 'Consolas';")
        lay.addWidget(self.lbl_status)

        row=QHBoxLayout()
        row.addWidget(self._small_btn("🗑  Clear",  self._clear))
        row.addWidget(self._small_btn("💾  Export", self._export_results))
        lay.addLayout(row)

        lay.addStretch()
        return panel

    def _reset_run_button(self):
        """Restores the run button to its vibrant, clickable state."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶   RUN DETECTION")
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #004080,stop:1 #006080);
                color:white; font:bold 16px 'Consolas'; border:1px solid {ACCENT};
                border-radius:8px; letter-spacing:1px;
            }}
            QPushButton:hover {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #005090,stop:1 #0070B0);
                border:1px solid #63B3ED;
            }}
        """)

    def _set_processing_button(self):
        """Changes the run button to a muted, processing state."""
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳   PROCESSING...")
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background:#4A5568; color:#E2E8F0; font:bold 16px 'Consolas'; 
                border:1px solid #2D3748; border-radius:8px; letter-spacing:2px;
            }}
        """)

    # ── Right Panel (tabs) ─────────────────────────────────────

    def _build_right_panel(self):
        panel=QFrame(); panel.setStyleSheet(f"background:{DARK_BG}; border-radius: 8px;")
        lay=QVBoxLayout(panel); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        tabs=QTabWidget()
        
        tabs.tabBar().setCursor(Qt.PointingHandCursor)
        
        # Enhanced floating tab design
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ background:{DARK_BG}; border:none; }}
            QTabBar::tab {{
                background:{PANEL_BG}; color:{TEXT_MUTED};
                padding:12px 20px; font:14px 'Consolas'; 
                min-width: 150px;
                border:none; border-right:1px solid {BORDER};
                border-bottom: 2px solid {BORDER};
            }}
            QTabBar::tab:selected {{ 
                background:{CARD_BG}; color:{ACCENT}; 
                border-bottom: 3px solid {ACCENT}; 
                font-weight:bold; 
            }}
            QTabBar::tab:hover:!selected {{ background:#F7FAFC; color:{TEXT_PRIMARY}; }}
        """)
        tabs.addTab(self._build_results_tab(),  "📊   Results")
        tabs.addTab(self._build_videos_tab(),   "🎬   Videos")
        tabs.addTab(self._build_audio_tab(),    "🎙   Audio Analysis")
        tabs.addTab(self._build_history_tab(),  "📋   History")
        tabs.addTab(self._build_about_tab(),    "ℹ    About")
        lay.addWidget(tabs)
        return panel

    # ── Results Tab ────────────────────────────────────────────

    def _build_results_tab(self):
        w=QWidget(); w.setStyleSheet(f"background:{PANEL_BG};")
        outer=QVBoxLayout(w); outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        # ── Bars (top) ──
        bars_w=QWidget(); bars_w.setStyleSheet(f"background:{PANEL_BG};")
        bars_lay=QVBoxLayout(bars_w); bars_lay.setContentsMargins(30,20,30,14); bars_lay.setSpacing(0)

        sec1=QLabel("MODALITY PROBABILITIES")
        sec1.setStyleSheet(
            f"color:{TEXT_MUTED}; font:11px 'Consolas'; letter-spacing:2px;"
            f"border-bottom:1px solid {BORDER}; padding-bottom:6px; margin-bottom:2px;")
        bars_lay.addWidget(sec1)

        self.bar_rgb  =ProbabilityBar("RGB · RCapsGRU_RGB",   "FrameDiff → 40 motion frames","#1A3A5C")
        self.bar_ir   =ProbabilityBar("IR · RCapsGRU_IR",     "DualMOG2 → 40 motion frames", "#276749")
        self.bar_audio=ProbabilityBar("Audio · AudioCNNGRU",  "4 s window, 2 s hop → mean P","#718096")
        for b in [self.bar_rgb,self.bar_ir,self.bar_audio]: bars_lay.addWidget(b)

        bars_lay.addSpacing(16)

        sec2=QLabel("FUSION RESULTS")
        sec2.setStyleSheet(
            f"color:{TEXT_MUTED}; font:11px 'Consolas'; letter-spacing:2px;"
            f"border-bottom:1px solid {BORDER}; padding-bottom:6px; margin-bottom:2px;")
        bars_lay.addWidget(sec2)

        self.bar_static =ProbabilityBar("Static Weighted",  "w_rgb=0.40   w_ir=0.35   w_audio=0.25","#276749")
        self.bar_dynamic=ProbabilityBar("Dynamic Weighted", "confidence = |P − 0.5| × 2",            "#B7791F")
        for b in [self.bar_static,self.bar_dynamic]: bars_lay.addWidget(b)

        note=QLabel(
            "PIPELINE REFERENCE   RGB: FrameDiff → RCapsGRU_RGB  ·  "
            "IR: DualMOG2 → RCapsGRU_IR  ·  Audio: sliding window → AudioCNNGRU  ·  τ = 0.50")
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color:{TEXT_MUTED}; font:10px 'Consolas';"
            f"border-top:1px solid {BORDER}; padding-top:8px; margin-top:10px;")
        bars_lay.addWidget(note)
        outer.addWidget(bars_w, stretch=3)

        # ── Separator ──
        sep=QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{BORDER}; max-height:1px;")
        outer.addWidget(sep)

        # ── Bottom: Gauge + Verdict ──
        bottom_w=QWidget(); bottom_w.setStyleSheet(f"background:{CARD_BG};")
        bottom_lay=QHBoxLayout(bottom_w)
        bottom_lay.setContentsMargins(30,18,30,18)
        bottom_lay.setSpacing(28)

        # Gauge column
        gauge_col=QVBoxLayout(); gauge_col.setSpacing(4)
        lbl_conf=QLabel("CONFIDENCE GAUGE  (0 = left  →  100 = right)")
        lbl_conf.setAlignment(Qt.AlignCenter)
        lbl_conf.setStyleSheet(
            f"color:{TEXT_MUTED}; font:11px 'Consolas'; letter-spacing:1px;")
        gauge_col.addWidget(lbl_conf)
        self.gauge=ConfidenceGauge()
        gauge_col.addWidget(self.gauge)
        bottom_lay.addLayout(gauge_col, stretch=3)

        # Verdict card
        self.verdict_card=VerdictCard()
        bottom_lay.addWidget(self.verdict_card, stretch=2)

        outer.addWidget(bottom_w, stretch=2)
        return w

    # ── Videos Tab ────────────────────────────────────────────

    def _build_videos_tab(self):
        w=QWidget(); w.setStyleSheet(f"background:{DARK_BG};")
        lay=QHBoxLayout(w); lay.setContentsMargins(14,14,14,14); lay.setSpacing(14)
        self.player_rgb=VideoPlayer("RGB OUTPUT — FrameDiff Tracker + Approach B Gated")
        self.player_ir =VideoPlayer("IR OUTPUT  — DualMOG2  Tracker + Approach B Gated")
        
        c1 = self._wrap_in_card(self.player_rgb); self._apply_shadow(c1)
        c2 = self._wrap_in_card(self.player_ir); self._apply_shadow(c2)
        
        lay.addWidget(c1)
        lay.addWidget(c2)
        return w

    # ── Audio Tab ─────────────────────────────────────────────

    def _build_audio_tab(self):
        w=QWidget(); w.setStyleSheet(f"background:{PANEL_BG};")
        lay=QVBoxLayout(w); lay.setContentsMargins(14,14,14,14); lay.setSpacing(10)
        info=QLabel(
            "  Audio analysis: 4-second sliding window, 2-second hop.  "
            "Green bars = Not Drone  |  Red bars = Drone Detected  |  Dashed line = Threshold (0.5)")
        info.setWordWrap(True)
        info.setStyleSheet(
            f"background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE;"
            f"border-radius:5px; font:11px Arial; padding:10px;")
        lay.addWidget(info)
        scroll=QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ background:{PANEL_BG}; border:1px solid {BORDER}; border-radius:6px; }}")
        self.audio_viz_label=QLabel()
        self.audio_viz_label.setAlignment(Qt.AlignCenter)
        self.audio_viz_label.setMinimumHeight(500)
        self.audio_viz_label.setStyleSheet(
            f"background:{CARD_BG}; color:{TEXT_MUTED}; font:13px Arial; border-radius:4px;")
        self.audio_viz_label.setText(
            "\n\n\n  No audio file processed yet.\n"
            "  Select an audio file and click Run Detection.\n\n\n")
        scroll.setWidget(self.audio_viz_label)
        lay.addWidget(scroll,stretch=1)
        self.audio_stats_bar=QLabel("Segments: —  |  Avg probability: —  |  Decision: —")
        self.audio_stats_bar.setStyleSheet(
            f"background:{CARD_BG}; color:{TEXT_MUTED}; font:11px Consolas;"
            f"padding:10px; border:1px solid {BORDER}; border-radius:5px;")
        lay.addWidget(self.audio_stats_bar)
        return w

    # ── History Tab ────────────────────────────────────────────

    def _build_history_tab(self):
        w=QWidget(); w.setStyleSheet(f"background:{DARK_BG};")
        lay=QVBoxLayout(w); lay.setContentsMargins(14,14,14,14)
        self.history_table=QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp","RGB File","IR File","P_RGB","P_IR","P_Audio","Static Fusion","Decision"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Zebra-striping enabled
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet(f"""
            QTableWidget {{
                background:{DARK_BG}; alternate-background-color:{PANEL_BG};
                color:{TEXT_PRIMARY}; gridline-color:{BORDER};
                font:12px 'Consolas'; border:1px solid {BORDER}; border-radius:5px;
            }}
            QHeaderView::section {{
                background:{CARD_BG}; color:{ACCENT}; font:bold 12px 'Consolas';
                padding:10px; border:none; border-bottom:2px solid {BORDER}; border-right:1px solid {BORDER};
            }}
            QTableWidget::item:selected {{ background:#EBF8FF; color:{ACCENT}; }}""")
            
        lay.addWidget(self.history_table)
        lay.addWidget(self._small_btn("🗑  Clear History", self._clear_history))
        return w

    # ── About Tab ─────────────────────────────────────────────

    def _build_about_tab(self):
        w=QWidget(); w.setStyleSheet(f"background:{DARK_BG};")
        lay=QVBoxLayout(w); lay.setContentsMargins(30,30,30,30)
        about=QTextEdit(); about.setReadOnly(True)
        about.setStyleSheet(
            f"background:{CARD_BG}; color:{TEXT_PRIMARY};"
            f"font:12px 'Consolas'; border:1px solid {BORDER}; border-radius:6px;")
        about.setHtml(f"""
        <div style="font-family:Consolas; padding:10px;">
        <h2 style="color:#2B6CB0; font-family:'Times New Roman';">◈ DRONE DETECTION SYSTEM</h2>
        <h3 style="color:#276749;">Approach B — Gated Inference Pipeline</h3><hr>
        <h4>PIPELINE ARCHITECTURE</h4>
        <pre>
  Input Video (RGB / IR / Audio)
        ↓
  [RGB]   FrameDiff  → Motion Frame Indices
  [IR]    DualMOG2   → Motion Frame Indices
  [Audio] Sliding Window (4s / 2s hop)
        ↓
  Sample 40 frames from MOTION indices only
        ↓
  RCapsGRU (RGB/IR)    AudioCNNGRU (Audio)
        ↓
  Static  Fusion (w_rgb=0.40, w_ir=0.35, w_audio=0.25)
  Dynamic Fusion (|P−0.5|×2  confidence weight)
        ↓
  Final Decision: DRONE / NOT DRONE  (τ = 0.50)
        </pre>
        <h4>TRACKER SELECTION</h4>
        <p>RGB → <b>FrameDiff</b> (Best F1=0.1290, Best IoU=0.1429)</p>
        <p>IR  → <b>DualMOG2</b>  (Best F1=0.0696, Lowest Fragmentation=0.07)</p>
        <h4>DATASET</h4>
        <pre>
  RGB: 257 videos × 4 methods = 1,028 evaluations
  IR:  340 videos × 4 methods = 1,360 evaluations
  Classes: Drone, Airplane, Bird, Helicopter
        </pre>
        <p>Device: {DEVICE}  |  Librosa: {'Available' if LIBROSA_OK else 'Not installed'}</p>
        </div>""")
        lay.addWidget(about)
        return w

    # ── Helper Builders ────────────────────────────────────────

    def _group(self, title):
        g=QGroupBox(title)
        g.setStyleSheet(f"""
            QGroupBox {{
                color:{ACCENT}; font:bold 14px 'Consolas';
                border:1px solid {BORDER}; border-radius:7px;
                margin-top:12px; padding-top:10px; background:{CARD_BG};
            }}
            QGroupBox::title {{
                subcontrol-origin:margin; subcontrol-position:top left;
                left:12px; padding:0 5px; background:{CARD_BG};
            }}""")
        return g

    def _wrap_in_card(self, w):
        card=QFrame()
        card.setStyleSheet(f"background:{CARD_BG}; border:1px solid {BORDER}; border-radius:8px;")
        l=QVBoxLayout(card); l.setContentsMargins(10,10,10,10); l.addWidget(w)
        return card

    def _action_btn(self, text, cb):
        b=QPushButton(text); b.clicked.connect(cb)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{PANEL_BG}; color:{ACCENT}; border:1px solid {ACCENT};
                border-radius:5px; padding:10px; font:bold 14px 'Consolas';
            }}
            QPushButton:hover {{ background:{CARD_BG}; }}""")
        return b

    def _file_btn(self, text, cb):
        b=QPushButton(text); b.clicked.connect(cb)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{DARK_BG}; color:{TEXT_PRIMARY}; border:1px solid {BORDER};
                border-radius:5px; padding:10px 12px; font:14px 'Consolas'; text-align:left;
            }}
            QPushButton:hover {{ background:{CARD_BG}; border-color:{ACCENT}; color:{ACCENT}; }}""")
        return b

    def _small_btn(self, text, cb):
        b=QPushButton(text); b.clicked.connect(cb)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(f"""
            QPushButton {{
                background:{DARK_BG}; color:{TEXT_MUTED}; border:1px solid {BORDER};
                border-radius:5px; padding:8px 14px; font:13px 'Consolas';
            }}
            QPushButton:hover {{ background:{CARD_BG}; color:{TEXT_PRIMARY}; border-color:{ACCENT}; }}""")
        return b

    def _path_label(self, text):
        l=QLabel(text); l.setWordWrap(True)
        l.setStyleSheet(f"color:{TEXT_MUTED}; font:12px 'Consolas'; padding:3px 10px;")
        return l

    def _apply_theme(self):
        # Applied sleek MacOS style scrollbars globally
        self.setStyleSheet(f"""
            QMainWindow {{ background:{DARK_BG}; }}
            QWidget {{ background:{DARK_BG}; color:{TEXT_PRIMARY}; }}
            QScrollBar:vertical {{
                background:transparent; width:10px; margin:0px; border-radius:5px;
            }}
            QScrollBar::handle:vertical {{
                background:#CBD5E0; min-height:30px; border-radius:5px;
            }}
            QScrollBar::handle:vertical:hover {{
                background:#A0AEC0;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height:0px;
            }}
            QScrollBar:horizontal {{
                background:transparent; height:10px; margin:0px; border-radius:5px;
            }}
            QScrollBar::handle:horizontal {{
                background:#CBD5E0; min-width:30px; border-radius:5px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background:#A0AEC0;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width:0px;
            }}
        """)

    def _setup_menu(self):
        mb=self.menuBar()
        mb.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_PRIMARY}; font:11px 'Consolas'; border-bottom:1px solid {BORDER};")
        file_menu=mb.addMenu("File")
        for name,cb in [
            ("Set Model Paths",      self._set_model_paths),
            ("Set Output Folder",    self._set_output_dir),
            ("Export Results (CSV)", self._export_results),
            ("Exit",                 self.close),
        ]:
            a=QAction(name,self); a.triggered.connect(cb); file_menu.addAction(a)
        help_menu=mb.addMenu("Help")
        a=QAction("About",self)
        a.triggered.connect(lambda: QMessageBox.information(
            self,"About","Drone Detection System\nApproach B — Gated Pipeline\nRGB: FrameDiff | IR: DualMOG2"))
        help_menu.addAction(a)

    # ── File Selection ─────────────────────────────────────────

    def _select_rgb(self):
        f,_=QFileDialog.getOpenFileName(self,"Select RGB Video","","Video (*.mp4 *.avi *.mov *.mkv)")
        if f:
            self.rgb_path=f; name=os.path.basename(f)
            self.lbl_rgb.setText(f"✓  {name}")
            self.lbl_rgb.setStyleSheet(f"color:{ACCENT2}; font:12px 'Consolas'; padding:3px 10px;")
            self.status_bar.showMessage(f"RGB: {name}")

    def _select_ir(self):
        f,_=QFileDialog.getOpenFileName(self,"Select IR Video","","Video (*.mp4 *.avi *.mov *.mkv)")
        if f:
            self.ir_path=f; name=os.path.basename(f)
            self.lbl_ir.setText(f"✓  {name}")
            self.lbl_ir.setStyleSheet(f"color:{ACCENT2}; font:12px 'Consolas'; padding:3px 10px;")
            self.status_bar.showMessage(f"IR: {name}")

    def _select_audio(self):
        f,_=QFileDialog.getOpenFileName(self,"Select Audio","","Audio (*.wav *.mp3)")
        if f:
            self.audio_path=f; name=os.path.basename(f)
            self.lbl_audio.setText(f"✓  {name}")
            self.lbl_audio.setStyleSheet(f"color:{ACCENT2}; font:12px 'Consolas'; padding:3px 10px;")

    # ── Model Loading ──────────────────────────────────────────

    def _load_models(self):
        loaded=[]
        try:
            if os.path.exists(self.RGB_MODEL_PATH):
                self.rgb_model=RCapsGRU_RGB().to(DEVICE)
                self.rgb_model.load_state_dict(torch.load(self.RGB_MODEL_PATH,map_location=DEVICE))
                self.rgb_model.eval(); loaded.append("RGB ✓")
        except Exception as e: loaded.append(f"RGB ✗ ({e})")
        try:
            if os.path.exists(self.IR_MODEL_PATH):
                self.ir_model=RCapsGRU_IR().to(DEVICE)
                self.ir_model.load_state_dict(torch.load(self.IR_MODEL_PATH,map_location=DEVICE))
                self.ir_model.eval(); loaded.append("IR ✓")
        except Exception as e: loaded.append(f"IR ✗ ({e})")
        try:
            if os.path.exists(self.AUDIO_MODEL_PATH) and LIBROSA_OK:
                self.audio_model=AudioCNNGRU().to(DEVICE)
                self.audio_model.load_state_dict(torch.load(self.AUDIO_MODEL_PATH,map_location=DEVICE))
                self.audio_model.eval(); loaded.append("Audio ✓")
        except Exception as e: loaded.append(f"Audio ✗ ({e})")

        rgb_s  ="RCapsGRU_RGB   loaded" if self.rgb_model   else "RCapsGRU_RGB   not found"
        ir_s   ="RCapsGRU_IR    loaded" if self.ir_model    else "RCapsGRU_IR    not found"
        audio_s="AudioCNNGRU    loaded" if self.audio_model else "AudioCNNGRU    not found"
        status ="  |  ".join(loaded) if loaded else "No models found"
        self.lbl_model_status.setText(f"{rgb_s}\n{ir_s}\n{audio_s}")
        self.lbl_model_status.setStyleSheet(
            f"color:{'#276749' if '✓' in status else DRONE_RED}; font:13px 'Consolas'; line-height:1.8;")
        self.status_bar.showMessage(f"Models: {status}")
        lnames=[]
        if self.rgb_model:   lnames.append("✓ RGB")
        if self.ir_model:    lnames.append("✓ IR")
        if self.audio_model: lnames.append("✓ Audio")
        self.lbl_header_models.setText("   ".join(lnames))

    # ── Run Detection ──────────────────────────────────────────

    def _run_detection(self):
        if not self.rgb_path and not self.ir_path and not self.audio_path:
            QMessageBox.warning(self,"No Input","Please select at least one input file."); return
        if not self.rgb_model and not self.ir_model and not self.audio_model:
            QMessageBox.warning(self,"No Models","Models not loaded. Click 'Load Models' first."); return
        
        self._set_processing_button() # Trigger the processing UI state
        self.progress.setValue(0)
        
        self.worker=InferenceWorker(
            self.rgb_path,self.ir_path,self.audio_path,
            self.rgb_model,self.ir_model,self.audio_model,self.output_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.video_ready.connect(self._on_video_ready)
        self.worker.start()

    def _on_progress(self, val, msg):
        self.progress.setValue(val); self.lbl_status.setText(msg); self.status_bar.showMessage(msg)

    def _on_result(self, results):
        self.results=results; 
        self._reset_run_button() # Restore the button
        
        p_rgb  =results.get('rgb')
        p_ir   =results.get('ir')
        p_audio=results.get('audio')
        p_stat =results.get('static_fusion')
        p_dyn  =results.get('dynamic_fusion')

        if p_rgb   is not None: self.bar_rgb.setValue(p_rgb)
        if p_ir    is not None: self.bar_ir.setValue(p_ir)
        if p_audio is not None: self.bar_audio.setValue(p_audio)
        if p_stat  is not None: self.bar_static.setValue(p_stat)
        if p_dyn   is not None: self.bar_dynamic.setValue(p_dyn)

        best = p_dyn
        best = (
                    p_dyn if p_dyn is not None else
                    p_stat if p_stat is not None else
                    p_rgb if p_rgb is not None else
                    p_ir if p_ir is not None else
                    p_audio if p_audio is not None else
                    0.0
                )
        self.gauge.setValue(best)

        is_drone=best>=0.5
        verdict ="Drone Detected" if is_drone else "Not Drone"
        rgb_m   =results.get('rgb_motion_frames',0)
        ir_m    =results.get('ir_motion_frames',0)
        ts      =results.get('timestamp','')
        info    =f"RGB motion frames: {rgb_m}\nIR motion frames: {ir_m}\n{ts}"
        self.verdict_card.update_result(verdict, best*100, info, is_drone)

        if 'audio_pixmap' in results:
            pix=results['audio_pixmap']
            self.audio_viz_label.setPixmap(
                pix.scaled(1100,750,Qt.KeepAspectRatio,Qt.SmoothTransformation))
            self.audio_viz_label.setMinimumSize(pix.width()//2,pix.height()//2)
            p_a=results.get('audio')
            if p_a is not None:
                decision="DRONE DETECTED" if p_a>=0.5 else "NOT DRONE"
                fname=os.path.basename(self.audio_path) if self.audio_path else "—"
                segs=len(results.get('audio_segs',[]))
                color=DRONE_RED if p_a>=0.5 else SAFE_GREEN
                self.audio_stats_bar.setText(
                    f"File: {fname}  |  Segments: {segs}  |  "
                    f"Avg probability: {p_a:.4f}  |  Decision: {decision}")
                self.audio_stats_bar.setStyleSheet(
                    f"background:{CARD_BG}; color:{color}; font:bold 11px Consolas;"
                    f"padding:10px; border:1px solid {color}; border-radius:5px;")

        self._add_to_history(results)
        rgb_str =f"{p_rgb:.3f}"  if p_rgb  is not None else "N/A"
        ir_str  =f"{p_ir:.3f}"   if p_ir   is not None else "N/A"
        fuse_str=f"{p_stat:.3f}" if p_stat is not None else "N/A"
        self.status_bar.showMessage(
            f"Detection complete —  RGB:{rgb_str}  IR:{ir_str}  "
            f"Fusion:{fuse_str}  →  {'DRONE' if is_drone else 'NOT DRONE'}")

    def _on_video_ready(self, modality, path):
        if modality=="RGB": self.player_rgb.load(path)
        elif modality=="IR": self.player_ir.load(path)

    def _on_error(self, msg):
        self._reset_run_button() # Restore the button
        QMessageBox.critical(self,"Error",f"Detection failed:\n{msg}")
        self.status_bar.showMessage(f"Error: {msg}")

    # ── History ────────────────────────────────────────────────

    def _add_to_history(self, r):
        self.history.append(r)
        row=self.history_table.rowCount(); self.history_table.insertRow(row)
        p_s=r.get('static_fusion'); is_drone=p_s is not None and p_s>=0.5
        vals=[
            r.get('timestamp',''),
            os.path.basename(self.rgb_path)   if self.rgb_path   else '—',
            os.path.basename(self.ir_path)    if self.ir_path    else '—',
            f"{r['rgb']:.4f}"   if r.get('rgb')   is not None else '—',
            f"{r['ir']:.4f}"    if r.get('ir')    is not None else '—',
            f"{r['audio']:.4f}" if r.get('audio') is not None else '—',
            f"{p_s:.4f}"        if p_s is not None else '—',
            "DRONE" if is_drone else "NOT DRONE",
        ]
        for ci,val in enumerate(vals):
            item=QTableWidgetItem(str(val)); item.setTextAlignment(Qt.AlignCenter)
            if ci==7: item.setForeground(QColor(DRONE_RED if is_drone else SAFE_GREEN))
            self.history_table.setItem(row,ci,item)

    def _clear_history(self):
        self.history_table.setRowCount(0); self.history.clear()

    # ── Export ─────────────────────────────────────────────────

    def _export_results(self):
        if not self.history:
            QMessageBox.information(self,"No Data","Run a detection first."); return
        path,_=QFileDialog.getSaveFileName(
            self,"Export Results","drone_detection_results.csv","CSV (*.csv)")
        if not path: return
        with open(path,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(["Timestamp","RGB File","IR File","P_RGB","P_IR","P_Audio",
                             "Static Fusion","Dynamic Fusion","Decision"])
            for r in self.history:
                p_s=r.get('static_fusion')
                writer.writerow([
                    r.get('timestamp',''),self.rgb_path,self.ir_path,
                    r.get('rgb',''),r.get('ir',''),r.get('audio',''),
                    r.get('static_fusion',''),r.get('dynamic_fusion',''),
                    "DRONE" if p_s and p_s>=0.5 else "NOT DRONE"])
        QMessageBox.information(self,"Exported",f"Results saved to:\n{path}")

    # ── Clear ──────────────────────────────────────────────────

    def _clear(self):
        self._reset_run_button()
        self.rgb_path=self.ir_path=self.audio_path=""
        for l in [self.lbl_rgb,self.lbl_ir,self.lbl_audio]:
            l.setText("No file selected")
            l.setStyleSheet(f"color:{TEXT_MUTED}; font:12px 'Consolas'; padding:3px 10px;")
        self.progress.setValue(0); self.lbl_status.setText("Cleared.")
        for bar in [self.bar_rgb,self.bar_ir,self.bar_audio,self.bar_static,self.bar_dynamic]:
            bar.reset()
        self.gauge.reset(); self.verdict_card.reset()
        for player in [self.player_rgb,self.player_ir]:
            if player.cap: player.cap.release(); player.cap=None
            player.timer.stop(); player.frame_lbl.clear(); player.frame_lbl.setText("No video loaded")
        self.audio_viz_label.clear()
        self.audio_viz_label.setText(
            "\n\n\n  No audio file processed yet.\n"
            "  Select an audio file and click Run Detection.\n\n\n")
        self.audio_stats_bar.setText("Segments: —  |  Avg probability: —  |  Decision: —")
        self.audio_stats_bar.setStyleSheet(
            f"background:{CARD_BG}; color:{TEXT_MUTED}; font:11px Consolas;"
            f"padding:10px; border:1px solid {BORDER}; border-radius:5px;")

    def _set_model_paths(self):
        QMessageBox.information(self,"Model Paths",
            "Open this script and update:\n\n"
            "  self.RGB_MODEL_PATH   = r'path\\to\\rcaps_gru_best.pth'\n"
            "  self.IR_MODEL_PATH    = r'path\\to\\rcaps_gru_ir_best.pth'\n"
            "  self.AUDIO_MODEL_PATH = r'path\\to\\audio_cnn_gru_binary_weighted.pth'")

    def _set_output_dir(self):
        d=QFileDialog.getExistingDirectory(self,"Select Output Folder")
        if d: self.output_dir=d; QMessageBox.information(self,"Output Folder",f"Output saved to:\n{d}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    window=DroneDetectionGUI()
    sys.exit(app.exec_())