import json
import math
import os
import sys
from pathlib import Path

# RTSP via TCP 
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QAction, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QPushButton, QSizePolicy, QToolBar, QWidget,
)

CONFIG_DIR = Path.home() / ".multicam_rtsp"
CONFIG_FILE = CONFIG_DIR / "config.json"
MAX_WIDTH = 1280
MIN_SLOTS = 4

_zombies = []


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


# ----------------------
# Thread que lê o RTSP
# ----------------------
class CameraWorker(QThread):
    frame = Signal(QImage)
    status = Signal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self._running = True

    def stop(self):
        self._running = False

    def _wait(self, seconds):
        for _ in range(int(seconds * 10)):
            if not self._running:
                return
            self.msleep(100)

    def _open(self):
        params = []
        for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
            if hasattr(cv2, name):
                params += [getattr(cv2, name), 5000]
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        except TypeError:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        while self._running:
            self.status.emit("Conectando...")
            cap = self._open()
            if cap is None:
                self.status.emit("Sem sinal - tentando novamente...")
                self._wait(3)
                continue

            fails, showing = 0, False
            while self._running:
                ok, img = cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= 15:
                        break
                    self.msleep(100)
                    continue
                fails = 0
                if not showing:
                    self.status.emit("")
                    showing = True

                h, w = img.shape[:2]
                if w > MAX_WIDTH:
                    img = cv2.resize(img, (MAX_WIDTH, int(h * MAX_WIDTH / w)),
                                     interpolation=cv2.INTER_AREA)
                    h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.frame.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())

            cap.release()
            if self._running:
                self.status.emit("Conexão perdida - reconectando...")
                self._wait(2)


class WindowButton(QPushButton):
    def __init__(self, kind, w=46, h=34, round_bg=False, parent=None):
        super().__init__(parent)
        self.kind, self.round_bg, self._hover = kind, round_bg, False
        self.setFixedSize(w, h)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        danger = self.kind == "close"
        bg = None
        if self._hover:
            bg = QColor(196, 43, 28) if danger else QColor(255, 255, 255, 45)
        elif self.round_bg:
            bg = QColor(0, 0, 0, 170)
        if bg:
            p.setPen(Qt.NoPen)
            p.setBrush(bg)
            if self.round_bg:
                p.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
            else:
                p.drawRect(self.rect())

        p.setPen(QPen(QColor(245, 245, 245), 1.6))
        p.setBrush(Qt.NoBrush)
        cx, cy, s = self.width() / 2, self.height() / 2, 5.0
        if self.kind == "min":
            p.drawLine(QPointF(cx - s, cy + 1), QPointF(cx + s, cy + 1))
        elif self.kind == "max":
            p.drawRect(QRectF(cx - s, cy - s, 2 * s, 2 * s))
        elif self.kind == "restore":
            p.drawRect(QRectF(cx - s, cy - s + 2.5, 2 * s - 2.5, 2 * s - 2.5))
            p.drawPolyline(QPolygonF([
                QPointF(cx - s + 2.5, cy - s + 2.5), QPointF(cx - s + 2.5, cy - s),
                QPointF(cx + s, cy - s), QPointF(cx + s, cy + s - 2.5),
                QPointF(cx + s - 2.5, cy + s - 2.5)]))
        else:  # close
            p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
            p.drawLine(QPointF(cx - s, cy + s), QPointF(cx + s, cy - s))


# ------------------
# Quadro da câmera
# ------------------
class CameraWidget(QWidget):
    double_clicked = Signal(object)
    add_requested = Signal(object)
    edit_requested = Signal(object)
    remove_requested = Signal(object)

    def __init__(self, index, cam=None):
        super().__init__()
        self.index, self.cam = index, cam
        self.image = None
        self.message = "Conectando..." if cam else ""
        self.worker = None
        self._hover = False
        self.setMinimumSize(160, 90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.ArrowCursor if cam else Qt.PointingHandCursor)

    # ---- conexão ----
    def start(self):
        if self.worker or not self.cam:
            return
        self.worker = CameraWorker(self.cam["url"])
        self.worker.frame.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.start()

    def stop(self):
        w, self.worker = self.worker, None
        if not w:
            return
        for sig, slot in ((w.frame, self._on_frame), (w.status, self._on_status)):
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        w.stop()
        if not w.wait(3000):
            _zombies.append(w)

    def set_camera(self, cam):
        self.stop()
        self.cam, self.image = cam, None
        self.message = "Conectando..." if cam else ""
        self.setCursor(Qt.ArrowCursor if cam else Qt.PointingHandCursor)
        self.start()
        self.update()

    def _on_frame(self, img):
        self.image = img
        self.update()

    def _on_status(self, text):
        self.message = text
        self.update()

    # ---- desenho ----
    def _label(self, p, text):
        p.setFont(QFont("Sans", 10, QFont.Bold))
        w = p.fontMetrics().horizontalAdvance(text) + 16
        p.fillRect(0, 0, w, 24, QColor(0, 0, 0, 150))
        p.setPen(Qt.white)
        p.drawText(8, 17, text)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.cam is None:
            self._paint_empty(p)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        if self.image is not None:
            scaled = self.image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawImage((self.width() - scaled.width()) // 2,
                        (self.height() - scaled.height()) // 2, scaled)
        if self.message:
            p.setPen(QColor(255, 200, 50))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
        self._label(p, self.cam["name"])
        p.setPen(QColor(60, 60, 60))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _paint_empty(self, p):
        r = self.rect()
        p.fillRect(r, QColor(26, 28, 34))
        p.setPen(QPen(QColor(120, 170, 220) if self._hover else QColor(80, 86, 98), 2, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r.adjusted(10, 10, -10, -10), 12, 12)

        cx, cy = r.center().x(), r.center().y() - 14
        s = max(14, min(r.width(), r.height()) // 12)
        p.setPen(QPen(QColor(170, 195, 225) if self._hover else QColor(150, 160, 176),
                      4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(cx - s, cy, cx + s, cy)
        p.drawLine(cx, cy - s, cx, cy + s)
        p.setFont(QFont("Sans", 11, QFont.Bold))
        p.setPen(QColor(200, 208, 220))
        p.drawText(QRect(0, cy + s + 10, r.width(), 24), Qt.AlignHCenter | Qt.AlignTop, "Câmera livre")
        p.setFont(QFont("Sans", 9))
        p.setPen(QColor(130, 138, 152))
        p.drawText(QRect(0, cy + s + 34, r.width(), 20), Qt.AlignHCenter | Qt.AlignTop,
                   "Clique para adicionar")
        self._label(p, f"Câmera {self.index + 1}")

    # ---- interação ----
    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if (self.cam is None and e.button() == Qt.LeftButton
                and self.rect().contains(e.position().toPoint())):
            self.add_requested.emit(self)
        else:
            super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.cam is not None and e.button() == Qt.LeftButton:
            self.double_clicked.emit(self)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        if self.cam is None:
            menu.addAction("Adicionar câmera...", lambda: self.add_requested.emit(self))
        else:
            menu.addAction("Ampliar / voltar", lambda: self.double_clicked.emit(self))
            menu.addAction("Editar...", lambda: self.edit_requested.emit(self))
            menu.addAction("Remover", lambda: self.remove_requested.emit(self))
        menu.exec(e.globalPos())


# ---------------------
# Diálogo de cadastro
# ---------------------
class CameraDialog(QDialog):
    def __init__(self, parent=None, cam=None):
        super().__init__(parent)
        self.setWindowTitle("Câmera")
        self.setMinimumWidth(480)
        form = QFormLayout(self)
        self.name = QLineEdit(cam["name"] if cam else "")
        self.name.setPlaceholderText("Ex.: Casa - Sala")
        self.url = QLineEdit(cam["url"] if cam else "")
        self.url.setPlaceholderText("rtsp://usuario:senha@ip/stream")
        form.addRow("Nome:", self.name)
        form.addRow("URL RTSP:", self.url)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _validate(self):
        if not self.name.text().strip() or not self.url.text().strip().lower().startswith("rtsp"):
            QMessageBox.warning(self, "Dados inválidos",
                                "Informe um nome e uma URL começando com rtsp://")
            return
        self.accept()

    def result_cam(self):
        return {"name": self.name.text().strip(), "url": self.url.text().strip()}


# ------------------
# Janela principal
# ------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MultiCam RTSP")
        self.resize(1200, 700)
        self.slots = [None] * MIN_SLOTS
        self.widgets = []
        self.zoomed = None
        self._pre_zoom = "full"
        self._win_actions = []
        self._load()

        self.central = QWidget()
        self.central.setObjectName("central")
        self.central.setStyleSheet("#central { background: #0e0f12; }")
        self.central.installEventFilter(self)
        self.grid = QGridLayout(self.central)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(4)
        self.setCentralWidget(self.central)

        
        self.zoom_btn = WindowButton("close", 44, 44, round_bg=True, parent=self.central)
        self.zoom_btn.setToolTip("Voltar para as câmeras (Esc)")
        self.zoom_btn.clicked.connect(self.exit_zoom)
        self.zoom_btn.hide()

        self._build_toolbar()

        esc = QAction(self)
        esc.setShortcut(QKeySequence("Esc"))
        esc.triggered.connect(self.exit_zoom)
        self.addAction(esc)

        for i, cam in enumerate(self.slots):
            self._make_widget(i, cam)
        self.layout_widgets()
        for w in self.widgets:
            w.start()
        self._sync_buttons()

    # ---- barra superior ----
    def _build_toolbar(self):
        self.toolbar = QToolBar("Principal")
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet(
            "QToolBar { background: #1b1d22; border: none; spacing: 4px; padding: 0; }"
            "QToolButton { color: #e8e8e8; padding: 7px 14px; border: none; }"
            "QToolButton:hover { background: #34373f; }")
        self.addToolBar(self.toolbar)

        add = QAction("+ Adicionar câmera", self)
        add.triggered.connect(self.add_camera)
        self.toolbar.addAction(add)

        full = QAction("Tela cheia (F11)", self)
        full.setShortcut(QKeySequence("F11"))
        full.triggered.connect(self.toggle_fullscreen)
        self.toolbar.addAction(full)
        self.addAction(full)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        
        btn_min = WindowButton("min")
        btn_min.setToolTip("Minimizar")
        btn_min.clicked.connect(self.showMinimized)
        btn_res = WindowButton("restore")
        btn_res.setToolTip("Sair da tela cheia (maximizar janela)")
        btn_res.clicked.connect(self.showMaximized)
        btn_close = WindowButton("close")
        btn_close.setToolTip("Fechar o programa")
        btn_close.clicked.connect(self.close)
        for b in (btn_min, btn_res, btn_close):
            self._win_actions.append(self.toolbar.addWidget(b))

    def _sync_buttons(self):
        show = self.isFullScreen()
        for a in self._win_actions:
            a.setVisible(show)

    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            self._sync_buttons()
        super().changeEvent(e)

    def eventFilter(self, obj, ev):
        if obj is self.central and ev.type() == QEvent.Resize:
            self._place_zoom_btn()
        return super().eventFilter(obj, ev)

    def _place_zoom_btn(self):
        self.zoom_btn.move(self.central.width() - self.zoom_btn.width() - 18, 18)
        self.zoom_btn.raise_()

    # ---- persistência ----
    def _load(self):
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            raw = data.get("slots", data.get("cameras", []))
            slots = [c if isinstance(c, dict) and c.get("url") else None for c in raw]
            self.slots = slots + [None] * (MIN_SLOTS - len(slots))
        except Exception:
            pass

    def _save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps({"slots": self.slots}, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    # ---- grade ----
    def _make_widget(self, index, cam):
        w = CameraWidget(index, cam)
        w.double_clicked.connect(self.toggle_zoom)
        w.add_requested.connect(self.add_at)
        w.edit_requested.connect(self.edit_camera)
        w.remove_requested.connect(self.remove_camera)
        self.widgets.append(w)
        return w

    def layout_widgets(self):
        for w in self.widgets:
            self.grid.removeWidget(w)
        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 0)
        for c in range(self.grid.columnCount()):
            self.grid.setColumnStretch(c, 0)

        visible = [self.zoomed] if self.zoomed else self.widgets
        n = len(visible)
        cols = 1 if self.zoomed else (2 if n <= 4 else math.ceil(math.sqrt(n)))
        for w in self.widgets:
            w.setVisible(w in visible)
        for i, w in enumerate(visible):
            self.grid.addWidget(w, i // cols, i % cols)
        for r in range((n + cols - 1) // cols):
            self.grid.setRowStretch(r, 1)
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)

    def _trim(self):
        """Remove slots livres sobrando no final (mantém sempre pelo menos 4)."""
        while len(self.slots) > MIN_SLOTS and self.slots[-1] is None:
            self.slots.pop()
            w = self.widgets.pop()
            w.stop()
            w.setParent(None)
            w.deleteLater()
        self.layout_widgets()

    # ---- zoom de uma câmera ----
    def toggle_zoom(self, widget):
        if self.zoomed:
            self.exit_zoom()
        else:
            self.enter_zoom(widget)

    def enter_zoom(self, widget):
        if self.zoomed or widget.cam is None:
            return
        self._pre_zoom = "full" if self.isFullScreen() else ("max" if self.isMaximized() else "normal")
        self.zoomed = widget
        self.toolbar.hide()
        self.layout_widgets()
        self.zoom_btn.show()
        if not self.isFullScreen():
            self.showFullScreen()
        self._place_zoom_btn()

    def exit_zoom(self):
        if not self.zoomed:
            return
        self.zoomed = None
        self.zoom_btn.hide()
        self.toolbar.show()
        self.layout_widgets()
        if self._pre_zoom == "max":
            self.showMaximized()
        elif self._pre_zoom == "normal":
            self.showNormal()

    # ---- ações ----
    def toggle_fullscreen(self):
        if self.zoomed:
            return
        self.showMaximized() if self.isFullScreen() else self.showFullScreen()

    def add_camera(self):
        idx = next((i for i, c in enumerate(self.slots) if c is None), None)
        if idx is None:
            self.slots.append(None)
            idx = len(self.slots) - 1
            self._make_widget(idx, None)
            self.layout_widgets()
        self.add_at(self.widgets[idx])

    def add_at(self, widget):
        dlg = CameraDialog(self)
        if dlg.exec():
            cam = dlg.result_cam()
            self.slots[widget.index] = cam
            self._save()
            widget.set_camera(cam)
        self._trim()

    def edit_camera(self, widget):
        dlg = CameraDialog(self, widget.cam)
        if dlg.exec():
            cam = dlg.result_cam()
            self.slots[widget.index] = cam
            self._save()
            widget.set_camera(cam)

    def remove_camera(self, widget):
        if QMessageBox.question(self, "Remover", f"Remover “{widget.cam['name']}”?") \
                != QMessageBox.Yes:
            return
        if self.zoomed is widget:
            self.exit_zoom()
        self.slots[widget.index] = None
        self._save()
        widget.set_camera(None)
        self._trim()

    def closeEvent(self, e):
        for w in self.widgets:
            w.stop()
        super().closeEvent(e)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("multicam.rtsp.app")
        except Exception:
            pass

    app = QApplication(sys.argv)
    icon_file = resource_path("")
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec())
