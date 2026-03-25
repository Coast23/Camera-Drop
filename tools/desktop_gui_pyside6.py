import ctypes
import os
import re
import subprocess
import sys

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAKE_CACHE_PATH = os.path.join(BUILD_DIR, "CMakeCache.txt")


APP_STYLE = """
QWidget {
    background-color: #f4f7fb;
    color: #1d2939;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #d0d7e2;
    border-radius: 10px;
    background: #ffffff;
    top: -1px;
}

QTabBar::tab {
    background: #e8edf5;
    border: 1px solid #d0d7e2;
    padding: 8px 14px;
    margin-right: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    color: #334155;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #0f172a;
    border-bottom-color: #ffffff;
    font-weight: 600;
}

QLineEdit {
    background: #ffffff;
    border: 1px solid #c8d2e1;
    border-radius: 8px;
    padding: 7px 10px;
}

QLineEdit:focus {
    border: 1px solid #3b82f6;
}

QPushButton {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background: #1d4ed8;
}

QPushButton:pressed {
    background: #1e40af;
}

QGroupBox {
    border: 1px solid #d0d7e2;
    border-radius: 10px;
    margin-top: 8px;
    background: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #334155;
    font-weight: 600;
}

QLabel {
    color: #334155;
}
"""


class ProcessWorker(QObject):
    line = Signal(str)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, cmd_args):
        super().__init__()
        self.cmd_args = list(cmd_args)

    def run(self):
        try:
            cmd_args = list(self.cmd_args)
            exe_name = cmd_args[0]
            if sys.platform == "win32" and not exe_name.endswith(".exe"):
                cmd_args[0] = exe_name + ".exe"

            exe_path = os.path.join(BUILD_DIR, cmd_args[0])
            if not os.path.exists(exe_path):
                found_path = None
                for root, _, files in os.walk(BUILD_DIR):
                    if cmd_args[0] in files:
                        found_path = os.path.join(root, cmd_args[0])
                        break
                if found_path:
                    exe_path = found_path
                else:
                    self.failed.emit(f"[Error] Executable not found at {exe_path}. Did you compile it?")
                    return

            cmd_args[0] = exe_path

            process = subprocess.Popen(
                cmd_args,
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            if process.stdout:
                for out_line in process.stdout:
                    self.line.emit(out_line.rstrip("\n"))

            process.wait()
            self.finished.emit(process.returncode)
        except Exception as exc:
            self.failed.emit(f"[Exception] {str(exc)}")


class TextFileEditorDialog(QDialog):
    def __init__(self, parent=None, initial_path=""):
        super().__init__(parent)
        self.setWindowTitle("Edit Input Text File")
        self.resize(820, 560)

        self.path_edit = QLineEdit(initial_path)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Type or paste text content here...")

        root = QVBoxLayout(self)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Text File:"))
        path_row.addWidget(self.path_edit, stretch=1)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_file)
        path_row.addWidget(btn_browse)

        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self.load_file)
        path_row.addWidget(btn_load)

        root.addLayout(path_row)
        root.addWidget(self.text_edit, stretch=1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self.save_file)
        action_row.addWidget(btn_save)

        btn_apply = QPushButton("Save and Use This Path")
        btn_apply.clicked.connect(self._save_and_accept)
        action_row.addWidget(btn_apply)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        action_row.addWidget(btn_close)

        root.addLayout(action_row)

        if initial_path and os.path.exists(initial_path):
            self.load_file()

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Text File")
        if file_path:
            self.path_edit.setText(file_path)
            self.load_file()

    def _read_text_with_fallback(self, file_path):
        encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
        last_error = None
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read(), enc
            except UnicodeDecodeError as exc:
                last_error = exc
        raise last_error if last_error else UnicodeDecodeError("unknown", b"", 0, 1, "decode failed")

    def load_file(self):
        file_path = self.path_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Please choose a text file path first.")
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "File does not exist.")
            return

        try:
            content, used_encoding = self._read_text_with_fallback(file_path)
            self.text_edit.setPlainText(content)
            if used_encoding != "utf-8":
                QMessageBox.information(
                    self,
                    "Encoding Notice",
                    f"Loaded with {used_encoding}. It will be saved as UTF-8 for stable Chinese text support.",
                )
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Cannot read file:\n{exc}")

    def save_file(self):
        file_path = self.path_edit.text().strip()
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Text File", filter="Text files (*.txt);;All files (*.*)")
            if not file_path:
                return False
            self.path_edit.setText(file_path)

        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(self.text_edit.toPlainText())
            QMessageBox.information(self, "Saved", "File saved as UTF-8.")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Cannot save file:\n{exc}")
            return False

    def _save_and_accept(self):
        if self.save_file():
            self.accept()

    def selected_path(self):
        return self.path_edit.text().strip()


class CameraDropGUI(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera-Drop GUI Tool (C++ Wrapper) [PySide6]")
        self.resize(700, 550)
        self.setStyleSheet(APP_STYLE)

        self._threads = []
        self._workers = []

        self.enc_input = QLineEdit()
        self.enc_output = QLineEdit()
        self.dec_input = QLineEdit()
        self.dec_output = QLineEdit()

        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.tab_encoder = QWidget()
        self.tab_decoder = QWidget()
        self.tab_tests = QWidget()

        self.tabs.addTab(self.tab_encoder, "File to Video Encoder")
        self.tabs.addTab(self.tab_decoder, "Video to File Decoder")
        self.tabs.addTab(self.tab_tests, "System Tests")

        self._build_encoder_tab()
        self._build_decoder_tab()
        self._build_tests_tab()

        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout(console_group)
        console_layout.setContentsMargins(10, 14, 10, 10)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #0f172a;"
            "color: #d1e7ff;"
            "border: 1px solid #1e293b;"
            "border-radius: 8px;"
            "font-family: Consolas, 'Courier New', monospace;"
            "font-size: 12px;"
        )
        console_layout.addWidget(self.console)
        root_layout.addWidget(console_group, stretch=1)

        self.log_signal.connect(self.log)

    def log(self, message):
        self.console.append(message)
        self.console.moveCursor(QTextCursor.End)

    def run_process_in_background(self, cmd_args):
        self.log(f">>> Running: {' '.join(cmd_args)}")

        thread = QThread(self)
        worker = ProcessWorker(cmd_args)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.line.connect(self.log_signal.emit)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.finished.connect(lambda: self._cleanup_worker(thread, worker))

        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    def _cleanup_worker(self, thread, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        if thread in self._threads:
            self._threads.remove(thread)
        worker.deleteLater()
        thread.deleteLater()

    def _on_worker_finished(self, code):
        self.log(f"<<< Process finished with exit code {code}\n")

    def _on_worker_failed(self, message):
        self.log(message + "\n")

    def _get_windows_dll_version(self, dll_path):
        if sys.platform != "win32":
            return None
        if not os.path.exists(dll_path):
            return None

        size = ctypes.windll.version.GetFileVersionInfoSizeW(dll_path, None)
        if size == 0:
            return None

        data = ctypes.create_string_buffer(size)
        ok = ctypes.windll.version.GetFileVersionInfoW(dll_path, 0, size, data)
        if not ok:
            return None

        ffi_ptr = ctypes.c_void_p()
        ffi_len = ctypes.c_uint()
        ok = ctypes.windll.version.VerQueryValueW(data, "\\", ctypes.byref(ffi_ptr), ctypes.byref(ffi_len))
        if not ok or not ffi_ptr.value:
            return None

        class VS_FIXEDFILEINFO(ctypes.Structure):
            _fields_ = [
                ("dwSignature", ctypes.c_uint32),
                ("dwStrucVersion", ctypes.c_uint32),
                ("dwFileVersionMS", ctypes.c_uint32),
                ("dwFileVersionLS", ctypes.c_uint32),
                ("dwProductVersionMS", ctypes.c_uint32),
                ("dwProductVersionLS", ctypes.c_uint32),
                ("dwFileFlagsMask", ctypes.c_uint32),
                ("dwFileFlags", ctypes.c_uint32),
                ("dwFileOS", ctypes.c_uint32),
                ("dwFileType", ctypes.c_uint32),
                ("dwFileSubtype", ctypes.c_uint32),
                ("dwFileDateMS", ctypes.c_uint32),
                ("dwFileDateLS", ctypes.c_uint32),
            ]

        ffi = ctypes.cast(ffi_ptr, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
        major = (ffi.dwProductVersionMS >> 16) & 0xFFFF
        minor = ffi.dwProductVersionMS & 0xFFFF
        build = (ffi.dwProductVersionLS >> 16) & 0xFFFF
        rev = ffi.dwProductVersionLS & 0xFFFF
        return f"{major}.{minor}.{build}.{rev}"

    def _parse_cmake_cache(self, cache_path):
        info = {
            "ONNXRUNTIME_INCLUDE_DIR": None,
            "ONNXRUNTIME_LIBRARY": None,
            "ONNXRUNTIME_DLL": None,
        }
        if not os.path.exists(cache_path):
            return info

        with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ONNXRUNTIME_INCLUDE_DIR:"):
                    info["ONNXRUNTIME_INCLUDE_DIR"] = line.split("=", 1)[-1]
                elif line.startswith("ONNXRUNTIME_LIBRARY:"):
                    info["ONNXRUNTIME_LIBRARY"] = line.split("=", 1)[-1]
                elif line.startswith("ONNXRUNTIME_DLL:"):
                    info["ONNXRUNTIME_DLL"] = line.split("=", 1)[-1]
        return info

    def _read_ort_api_version_from_header(self, include_dir):
        if not include_dir:
            return None

        header_path = os.path.join(include_dir, "onnxruntime_c_api.h")
        if not os.path.exists(header_path):
            return None

        pattern = re.compile(r"#define\s+ORT_API_VERSION\s+(\d+)")
        with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    return int(match.group(1))
        return None

    def check_onnxruntime_environment(self):
        self.log("=== ONNX Runtime Environment Check ===")
        cache = self._parse_cmake_cache(CMAKE_CACHE_PATH)

        include_dir = cache.get("ONNXRUNTIME_INCLUDE_DIR")
        lib_path = cache.get("ONNXRUNTIME_LIBRARY")
        cmake_dll = cache.get("ONNXRUNTIME_DLL")
        build_dll = os.path.join(BUILD_DIR, "onnxruntime.dll")
        system_dll = os.path.join(os.environ.get("SystemRoot", "C:/Windows"), "System32", "onnxruntime.dll")

        self.log(f"Project root: {ROOT_DIR}")
        self.log(f"Build dir:    {BUILD_DIR}")
        self.log(f"CMake cache:  {CMAKE_CACHE_PATH}")
        self.log(f"ORT include:  {include_dir}")
        self.log(f"ORT lib:      {lib_path}")
        self.log(f"ORT dll(cfg): {cmake_dll}")
        self.log(f"ORT dll(build): {build_dll}")

        api_ver = self._read_ort_api_version_from_header(include_dir)
        if api_ver is None:
            self.log("ORT API version (from header): <unknown>")
        else:
            self.log(f"ORT API version (from header): {api_ver}")

        build_dll_ver = self._get_windows_dll_version(build_dll) if os.path.exists(build_dll) else None
        cfg_dll_ver = self._get_windows_dll_version(cmake_dll) if cmake_dll and os.path.exists(cmake_dll) else None
        sys_dll_ver = self._get_windows_dll_version(system_dll) if os.path.exists(system_dll) else None

        self.log(f"ORT version (build dll): {build_dll_ver or '<missing or unreadable>'}")
        self.log(f"ORT version (cfg dll):   {cfg_dll_ver or '<missing or unreadable>'}")
        self.log(f"ORT version (system32):  {sys_dll_ver or '<missing or unreadable>'}")

        warnings = []
        if cmake_dll and os.path.normcase(os.path.abspath(cmake_dll)) == os.path.normcase(os.path.abspath(system_dll)):
            warnings.append("CMake is pointing to System32/onnxruntime.dll. This may cause header/runtime mismatch.")

        if cmake_dll and not os.path.exists(cmake_dll):
            warnings.append("Configured ONNX Runtime DLL does not exist.")

        if not os.path.exists(build_dll):
            warnings.append("Build directory ONNX Runtime DLL is missing.")

        if cmake_dll and os.path.exists(cmake_dll) and os.path.exists(build_dll):
            if cfg_dll_ver and build_dll_ver and cfg_dll_ver != build_dll_ver:
                warnings.append("Configured DLL version and build DLL version are different.")

        if api_ver is not None and build_dll_ver:
            parts = build_dll_ver.split(".")
            if len(parts) >= 2:
                try:
                    major = int(parts[0])
                    minor = int(parts[1])
                    if major == 1 and api_ver != minor:
                        warnings.append(
                            f"Likely API mismatch: header ORT_API_VERSION={api_ver}, build dll looks like ORT {major}.{minor}."
                        )
                except ValueError:
                    pass

        if warnings:
            self.log("[WARN] Potential problems detected:")
            for warning in warnings:
                self.log(f"  - {warning}")
            QMessageBox.warning(self, "ONNX Runtime Check", "Environment check found potential mismatch issues. See console output.")
        else:
            self.log("[OK] No obvious ONNX Runtime mismatch detected.")
            QMessageBox.information(self, "ONNX Runtime Check", "Environment check passed.")

        self.log("=== End Environment Check ===\n")

    def _build_encoder_tab(self):
        layout = QGridLayout(self.tab_encoder)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(10)

        layout.addWidget(QLabel("Input File:"), 0, 0)
        layout.addWidget(self.enc_input, 0, 1)
        btn_input = QPushButton("Browse...")
        btn_input.clicked.connect(self._browse_encoder_input)
        layout.addWidget(btn_input, 0, 2)
        btn_edit_text = QPushButton("Edit Text...")
        btn_edit_text.clicked.connect(self._open_text_editor)
        layout.addWidget(btn_edit_text, 0, 3)

        layout.addWidget(QLabel("Output Video (*.mp4):"), 1, 0)
        layout.addWidget(self.enc_output, 1, 1)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self._browse_encoder_output)
        layout.addWidget(btn_output, 1, 2)

        run_btn = QPushButton("Run Encoder")
        run_btn.clicked.connect(self.run_encoder)

        run_row = QHBoxLayout()
        run_row.addStretch(1)
        run_row.addWidget(run_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row, 2, 0, 1, 3)

        layout.setColumnStretch(1, 1)

    def _open_text_editor(self):
        dialog = TextFileEditorDialog(self, self.enc_input.text().strip())
        if dialog.exec() == QDialog.Accepted:
            selected = dialog.selected_path()
            if selected:
                self.enc_input.setText(selected)

    def _build_decoder_tab(self):
        layout = QGridLayout(self.tab_decoder)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(10)

        layout.addWidget(QLabel("Input Video (*.mp4):"), 0, 0)
        layout.addWidget(self.dec_input, 0, 1)
        btn_input = QPushButton("Browse...")
        btn_input.clicked.connect(self._browse_decoder_input)
        layout.addWidget(btn_input, 0, 2)

        layout.addWidget(QLabel("Decoded File:"), 1, 0)
        layout.addWidget(self.dec_output, 1, 1)
        btn_output = QPushButton("Browse...")
        btn_output.clicked.connect(self._browse_decoder_output)
        layout.addWidget(btn_output, 1, 2)

        run_btn = QPushButton("Run Decoder")
        run_btn.clicked.connect(self.run_decoder)

        run_row = QHBoxLayout()
        run_row.addStretch(1)
        run_row.addWidget(run_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row, 2, 0, 1, 3)

        layout.setColumnStretch(1, 1)

    def _build_tests_tab(self):
        layout = QVBoxLayout(self.tab_tests)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        info = QLabel("Run inner C++ tests measuring encode/decode accuracy.")
        info.setAlignment(Qt.AlignLeft)
        layout.addWidget(info)

        btn_config = QPushButton("Run Config Accuracy Test")
        btn_config.clicked.connect(lambda: self.run_process_in_background(["config_acc_test"]))
        layout.addWidget(btn_config)

        btn_unit = QPushButton("Run Unit Tests")
        btn_unit.clicked.connect(lambda: self.run_process_in_background(["unit_test"]))
        layout.addWidget(btn_unit)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        btn_check = QPushButton("Check ONNX Runtime Environment")
        btn_check.clicked.connect(self.check_onnxruntime_environment)
        layout.addWidget(btn_check)

        layout.addStretch(1)

    def _browse_encoder_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if file_path:
            self.enc_input.setText(file_path)

    def _browse_encoder_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output Video", filter="MP4 files (*.mp4);;All files (*.*)")
        if file_path:
            self.enc_output.setText(file_path)

    def _browse_decoder_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Video")
        if file_path:
            self.dec_input.setText(file_path)

    def _browse_decoder_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Decoded File")
        if file_path:
            self.dec_output.setText(file_path)

    def run_encoder(self):
        infile = self.enc_input.text().strip()
        outfile = self.enc_output.text().strip()
        if not infile:
            QMessageBox.warning(self, "Warning", "Please select an input file.")
            return

        cmd = ["file_video_encoder", "--input", infile]
        if outfile:
            cmd.extend(["--video-out", outfile])
        self.run_process_in_background(cmd)

    def run_decoder(self):
        infile = self.dec_input.text().strip()
        outfile = self.dec_output.text().strip()
        if not infile:
            QMessageBox.warning(self, "Warning", "Please select an input video.")
            return

        cmd = ["file_video_decoder", "--input", infile]
        if outfile:
            cmd.extend(["--output", outfile])
        self.run_process_in_background(cmd)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraDropGUI()
    window.show()
    sys.exit(app.exec())
