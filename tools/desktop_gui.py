import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import sys
import ctypes
import re

# Constants for Paths
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAKE_CACHE_PATH = os.path.join(BUILD_DIR, "CMakeCache.txt")

class CameraDropGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera-Drop GUI Tool (C++ Wrapper)")
        self.geometry("700x550")
        
        # Configure layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # --- Top notebook for different tools ---
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Tab 1: Video Encoder
        self.tab_encoder = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_encoder, text="File to Video Encoder")
        self.build_encoder_tab()
        
        # Tab 2: Video Decoder
        self.tab_decoder = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_decoder, text="Video to File Decoder")
        self.build_decoder_tab()
        
        # Tab 3: System Tests
        self.tab_tests = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tests, text="System Tests")
        self.build_test_tab()
        
        # --- Bottom Console Output ---
        console_frame = ttk.LabelFrame(self, text="Console Output")
        console_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        console_frame.grid_rowconfigure(0, weight=1)
        console_frame.grid_columnconfigure(0, weight=1)
        
        self.console = tk.Text(console_frame, bg="black", fg="white", font=("Consolas", 9), state=tk.DISABLED)
        self.console.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(console_frame, command=self.console.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console.config(yscrollcommand=scrollbar.set)
        
    def log(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
        
    def run_process_in_background(self, cmd_args):
        self.log(f">>> Running: {' '.join(cmd_args)}")
        
        def task():
            try:
                # Determine executable with `.exe` if on windows, or directly on linux
                exe_name = cmd_args[0]
                if sys.platform == "win32" and not exe_name.endswith(".exe"):
                    cmd_args[0] = exe_name + ".exe"

                exe_path = os.path.join(BUILD_DIR, cmd_args[0])
                if not os.path.exists(exe_path):
                    # Some targets (e.g. wirehair unit_test) are placed in subdirectories.
                    found_path = None
                    for root, _, files in os.walk(BUILD_DIR):
                        if cmd_args[0] in files:
                            found_path = os.path.join(root, cmd_args[0])
                            break
                    if found_path:
                        exe_path = found_path
                    else:
                        self.log(f"[Error] Executable not found at {exe_path}. Did you compile it?")
                        return
                
                cmd_args[0] = exe_path
                
                # Run the process
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
                for line in process.stdout:
                    self.log(line.strip())
                process.wait()
                self.log(f"<<< Process finished with exit code {process.returncode}\n")
            except Exception as e:
                self.log(f"[Exception] {str(e)}\n")
                
        threading.Thread(target=task, daemon=True).start()

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
                m = pattern.search(line)
                if m:
                    return int(m.group(1))
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
            for w in warnings:
                self.log(f"  - {w}")
            messagebox.showwarning("ONNX Runtime Check", "Environment check found potential mismatch issues. See console output.")
        else:
            self.log("[OK] No obvious ONNX Runtime mismatch detected.")
            messagebox.showinfo("ONNX Runtime Check", "Environment check passed.")

        self.log("=== End Environment Check ===\n")

    # --- ENCODER TAB ---
    def build_encoder_tab(self):
        self.enc_input_var = tk.StringVar()
        self.enc_outdir_var = tk.StringVar()
        
        ttk.Label(self.tab_encoder, text="Input File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.tab_encoder, textvariable=self.enc_input_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.tab_encoder, text="Browse...", command=lambda: self.enc_input_var.set(filedialog.askopenfilename())).grid(row=0, column=2, padx=5)
        
        ttk.Label(self.tab_encoder, text="Output Video (*.mp4):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.tab_encoder, textvariable=self.enc_outdir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.tab_encoder, text="Browse...", command=lambda: self.enc_outdir_var.set(filedialog.asksaveasfilename(defaultextension=".mp4"))).grid(row=1, column=2, padx=5)
        
        ttk.Button(self.tab_encoder, text="Run Encoder", command=self.run_encoder).grid(row=2, column=1, pady=15)
        
    def run_encoder(self):
        infile = self.enc_input_var.get()
        outfile = self.enc_outdir_var.get()
        if not infile:
            messagebox.showwarning("Warning", "Please select an input file.")
            return
            
        cmd = ["file_video_encoder", "--input", infile]
        if outfile:
            cmd.extend(["--video-out", outfile])
        self.run_process_in_background(cmd)
        
    # --- DECODER TAB ---
    def build_decoder_tab(self):
        self.dec_input_var = tk.StringVar()
        self.dec_output_var = tk.StringVar()
        
        ttk.Label(self.tab_decoder, text="Input Video (*.mp4):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.tab_decoder, textvariable=self.dec_input_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.tab_decoder, text="Browse...", command=lambda: self.dec_input_var.set(filedialog.askopenfilename())).grid(row=0, column=2, padx=5)
        
        ttk.Label(self.tab_decoder, text="Decoded File:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.tab_decoder, textvariable=self.dec_output_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.tab_decoder, text="Browse...", command=lambda: self.dec_output_var.set(filedialog.asksaveasfilename())).grid(row=1, column=2, padx=5)
        
        ttk.Button(self.tab_decoder, text="Run Decoder", command=self.run_decoder).grid(row=2, column=1, pady=15)
        
    def run_decoder(self):
        infile = self.dec_input_var.get()
        outfile = self.dec_output_var.get()
        if not infile:
            messagebox.showwarning("Warning", "Please select an input video.")
            return
            
        cmd = ["file_video_decoder", "--input", infile]
        if outfile:
            cmd.extend(["--output", outfile])
        self.run_process_in_background(cmd)

    # --- TEST TAB ---
    def build_test_tab(self):
        ttk.Label(self.tab_tests, text="Run inner C++ tests measuring encode/decode accuracy.").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(self.tab_tests, text="Run Config Accuracy Test", command=lambda: self.run_process_in_background(["config_acc_test"])).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ttk.Button(self.tab_tests, text="Run Unit Tests", command=lambda: self.run_process_in_background(["unit_test"])).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Separator(self.tab_tests, orient="horizontal").grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        ttk.Button(self.tab_tests, text="Check ONNX Runtime Environment", command=self.check_onnxruntime_environment).grid(row=4, column=0, padx=10, pady=5, sticky="w")


if __name__ == "__main__":
    app = CameraDropGUI()
    app.mainloop()
