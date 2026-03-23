import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import sys

# Constants for Paths
BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")

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
                    self.log(f"[Error] Executable not found at {exe_path}. Did you compile it?")
                    return
                
                cmd_args[0] = exe_path
                
                # Run the process
                process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    self.log(line.strip())
                process.wait()
                self.log(f"<<< Process finished with exit code {process.returncode}\n")
            except Exception as e:
                self.log(f"[Exception] {str(e)}\n")
                
        threading.Thread(target=task, daemon=True).start()

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


if __name__ == "__main__":
    app = CameraDropGUI()
    app.mainloop()
