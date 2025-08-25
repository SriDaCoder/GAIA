# gaia_module.py
import os
import time
import threading
from collections import deque

# Optional system stats
try:
    import psutil
except Exception:
    psutil = None

# UI / plotting
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ML
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Synthetic environment
# -----------------------------
class SyntheticEnv:
    def __init__(self, f_soil=4, f_water=3, f_weather=5, action_dim=6):
        self.f_soil, self.f_water, self.f_weather = f_soil, f_water, f_weather
        self.action_dim = action_dim

    def sample_batch(self, batch_size=8, device="cpu"):
        soil = torch.randn(batch_size, self.f_soil, device=device)
        water = torch.randn(batch_size, self.f_water, device=device)
        weather = torch.randn(batch_size, self.f_weather, device=device)
        obs = {"soil": soil, "water": water, "weather": weather}
        labels = {
            "soil": soil.mean(dim=1, keepdim=True),
            "water": water.mean(dim=1, keepdim=True),
            "weather": weather.mean(dim=1, keepdim=True),
            "action": torch.randint(0, self.action_dim, (batch_size,), device=device),
            "risk": torch.rand(batch_size, device=device),
        }
        return obs, labels

# -----------------------------
# Subsystems / model
# -----------------------------
class Subsystem(nn.Module):
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))
    def forward(self, x): return self.net(x)

class DecisionHead(nn.Module):
    def __init__(self, in_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 32)
        self.action = nn.Linear(32, action_dim)
        self.provenance = nn.Linear(32, action_dim)
        self.risk = nn.Linear(32, 1)
    def forward(self, fused):
        h = torch.relu(self.fc(fused))
        return {
            "chosen": self.action(h),
            "provenance": torch.sigmoid(self.provenance(h)),
            "risk": torch.sigmoid(self.risk(h)),
        }

class Gaia(nn.Module):
    def __init__(self, f_soil, f_water, f_weather, action_dim):
        super().__init__()
        self.soil, self.water, self.weather = Subsystem(f_soil), Subsystem(f_water), Subsystem(f_weather)
        self.fuse = nn.Linear(8*3, 16)
        self.head = DecisionHead(16, action_dim)
    def forward(self, obs):
        s, w, we = self.soil(obs["soil"]), self.water(obs["water"]), self.weather(obs["weather"])
        fused = torch.relu(self.fuse(torch.cat([s, w, we], dim=1)))
        return self.head(fused), {"soil": s, "water": w, "weather": we}

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, outputs, labels):
        dec, subs = outputs
        loss_soil = self.mse(subs["soil"].mean(dim=1, keepdim=True), labels["soil"])
        loss_water = self.mse(subs["water"].mean(dim=1, keepdim=True), labels["water"])
        loss_weather = self.mse(subs["weather"].mean(dim=1, keepdim=True), labels["weather"])
        loss_action = self.ce(dec["chosen"], labels["action"])
        loss_risk = self.mse(dec["risk"].squeeze(), labels["risk"])
        total = loss_soil + loss_water + loss_weather + loss_action + loss_risk
        return total, {
            "soil": loss_soil, "water": loss_water, "weather": loss_weather,
            "action": loss_action, "risk": loss_risk
        }

# -----------------------------
# Trainer (importable + GUI uses)
# -----------------------------
class GaiaTrainer:
    def __init__(self, device="cpu", batch_size=16, lr=3e-4, ckpt="gaia_weights.pth", stop_flag_path="stop.flag"):
        self.env = SyntheticEnv()
        self.model = Gaia(self.env.f_soil, self.env.f_water, self.env.f_weather, self.env.action_dim).to(device)
        self.crit = MultiTaskLoss()
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.batch_size = batch_size
        self.ckpt = ckpt
        self.stop_flag_path = stop_flag_path

        self.step = 0
        self.running = False
        self.loss_history = deque(maxlen=500)
        self.sub_losses = {k: deque(maxlen=500) for k in ["soil","water","weather","action","risk"]}
        self.last_logs = {}
        self.last_loss = None

        if os.path.exists(self.ckpt):
            try:
                self.model.load_state_dict(torch.load(self.ckpt, map_location=device))
                print(f"Loaded weights from {self.ckpt}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    def _should_stop_external(self) -> bool:
        return os.path.exists(self.stop_flag_path)

    def train_step(self):
        obs, labels = self.env.sample_batch(self.batch_size, self.device)
        self.model.train()
        dec, subs = self.model(obs)
        loss, logs = self.crit((dec, subs), labels)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step += 1
        self.last_loss = float(loss.item())
        self.last_logs = {k: float(v.item()) for k,v in logs.items()}
        self.loss_history.append(self.last_loss)
        for k,v in self.last_logs.items():
            self.sub_losses[k].append(v)

        # periodic save
        if self.step % 100 == 0:
            try:
                torch.save(self.model.state_dict(), self.ckpt)
            except Exception as e:
                print(f"Save failed: {e}")

        return self.last_loss, self.last_logs, dec

    def save_now(self):
        try:
            torch.save(self.model.state_dict(), self.ckpt)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False

    def train_forever(self, sleep_s=0.05):
        self.running = True
        while self.running:
            if self._should_stop_external():
                # graceful: save and stop
                self.save_now()
                self.running = False
                break
            self.train_step()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def stop(self, save=True):
        self.running = False
        if save:
            self.save_now()

# -----------------------------
# Tkinter App with Tabs, Checkboxes, System Stats
# -----------------------------
class GaiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GAIA Trainer")
        self.trainer = GaiaTrainer()
        self.thread = None

        # Notebook (tabs)
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        # --- Training tab ---
        self.train_tab = ttk.Frame(self.nb)
        self.nb.add(self.train_tab, text="Training")

        top_box = ttk.Frame(self.train_tab)
        top_box.pack(fill="x", padx=8, pady=6)

        self.loss_lbl = ttk.Label(top_box, text="Loss: --", font=("Arial", 12))
        self.loss_lbl.pack(side="left", padx=5)

        self.step_lbl = ttk.Label(top_box, text="Step: 0", font=("Arial", 12))
        self.step_lbl.pack(side="left", padx=10)

        self.status_lbl = ttk.Label(top_box, text="Status: Idle", font=("Arial", 10))
        self.status_lbl.pack(side="left", padx=10)

        btn_box = ttk.Frame(self.train_tab)
        btn_box.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn_box, text="Start", command=self.start_training).pack(side="left", padx=4)
        ttk.Button(btn_box, text="Stop (save)", command=self.stop_training_save).pack(side="left", padx=4)
        ttk.Button(btn_box, text="Stop (no save)", command=self.stop_training_nosave).pack(side="left", padx=4)
        ttk.Button(btn_box, text="Save Now", command=self.save_now).pack(side="left", padx=4)

        # Checkpoint path + external stop flag path
        path_box = ttk.Frame(self.train_tab)
        path_box.pack(fill="x", padx=8, pady=6)
        ttk.Label(path_box, text="Checkpoint:").pack(side="left")
        self.ckpt_var = tk.StringVar(value=self.trainer.ckpt)
        ttk.Entry(path_box, textvariable=self.ckpt_var, width=30).pack(side="left", padx=4)
        ttk.Button(path_box, text="Browse", command=self.browse_ckpt).pack(side="left", padx=4)

        ttk.Label(path_box, text="Stop Flag:").pack(side="left", padx=(12,0))
        self.stopflag_var = tk.StringVar(value=self.trainer.stop_flag_path)
        ttk.Entry(path_box, textvariable=self.stopflag_var, width=20).pack(side="left", padx=4)
        ttk.Button(path_box, text="Set", command=self.apply_paths).pack(side="left", padx=4)
        ttk.Button(path_box, text="Create stop.flag", command=self.create_stopflag).pack(side="left", padx=4)

        # Loss selection checkboxes
        checks_frame = ttk.LabelFrame(self.train_tab, text="Show Sub-losses")
        checks_frame.pack(fill="x", padx=8, pady=6)
        self.loss_keys = ["soil","water","weather","action","risk"]
        self.loss_show = {k: tk.BooleanVar(value=True if k in ("soil","water","weather") else False) for k in self.loss_keys}
        for k in self.loss_keys:
            ttk.Checkbutton(checks_frame, text=k.capitalize(), variable=self.loss_show[k], command=self.redraw_plots).pack(side="left", padx=6)

        # Hyperparams
        hbox = ttk.LabelFrame(self.train_tab, text="Hyperparameters")
        hbox.pack(fill="x", padx=8, pady=6)
        ttk.Label(hbox, text="Batch Size").pack(side="left")
        self.bs_var = tk.StringVar(value=str(self.trainer.batch_size))
        ttk.Entry(hbox, textvariable=self.bs_var, width=6).pack(side="left", padx=4)
        ttk.Label(hbox, text="LR").pack(side="left", padx=(12,0))
        self.lr_var = tk.StringVar(value="0.0003")
        ttk.Entry(hbox, textvariable=self.lr_var, width=8).pack(side="left", padx=4)
        ttk.Button(hbox, text="Apply", command=self.apply_hparams).pack(side="left", padx=6)

        # Figure
        self.fig = Figure(figsize=(7,5), dpi=100)
        self.ax_total = self.fig.add_subplot(211)
        self.ax_sub = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.train_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)

        # --- System tab ---
        self.sys_tab = ttk.Frame(self.nb)
        self.nb.add(self.sys_tab, text="System")

        sys_box = ttk.Frame(self.sys_tab)
        sys_box.pack(fill="x", padx=8, pady=8)

        self.cpu_lbl = ttk.Label(sys_box, text="CPU: -- %", font=("Arial", 12))
        self.cpu_lbl.pack(side="left", padx=10)
        self.mem_lbl = ttk.Label(sys_box, text="Memory: -- %", font=("Arial", 12))
        self.mem_lbl.pack(side="left", padx=10)

        # per-process (if psutil available)
        self.proc_cpu_lbl = ttk.Label(self.sys_tab, text="Process CPU: -- %")
        self.proc_mem_lbl = ttk.Label(self.sys_tab, text="Process RSS: -- MB")
        self.proc_cpu_lbl.pack(anchor="w", padx=8, pady=(6,0))
        self.proc_mem_lbl.pack(anchor="w", padx=8, pady=(0,8))

        # schedule UI updates
        self.update_gui()
        self.update_sys()

        # window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- UI actions ----
    def start_training(self):
        if self.thread and self.thread.is_alive():
            return
        # ensure latest paths applied
        self.apply_paths()
        self.thread = threading.Thread(target=self.trainer.train_forever, kwargs={"sleep_s":0.05}, daemon=True)
        self.thread.start()
        self.status_lbl.config(text="Status: Running")

    def stop_training_save(self):
        self.trainer.stop(save=True)
        self.status_lbl.config(text="Status: Stopping (saving)...")

    def stop_training_nosave(self):
        self.trainer.stop(save=False)
        self.status_lbl.config(text="Status: Stopping...")

    def save_now(self):
        ok = self.trainer.save_now()
        messagebox.showinfo("Save", "Saved." if ok else "Save failed (see console).")

    def browse_ckpt(self):
        path = filedialog.asksaveasfilename(defaultextension=".pth", initialfile=os.path.basename(self.trainer.ckpt))
        if path:
            self.ckpt_var.set(path)

    def apply_paths(self):
        self.trainer.ckpt = self.ckpt_var.get().strip() or "gaia_weights.pth"
        self.trainer.stop_flag_path = self.stopflag_var.get().strip() or "stop.flag"

    def create_stopflag(self):
        try:
            with open(self.stopflag_var.get().strip() or "stop.flag", "w") as f:
                f.write("1")
            messagebox.showinfo("Stop Flag", "stop.flag created; training will stop soon.")
        except Exception as e:
            messagebox.showerror("Stop Flag", str(e))

    def apply_hparams(self):
        try:
            bs = int(self.bs_var.get())
            lr = float(self.lr_var.get())
            self.trainer.batch_size = bs
            for g in self.trainer.opt.param_groups:
                g["lr"] = lr
        except Exception as e:
            messagebox.showerror("Hyperparameters", f"Invalid values: {e}")

    # ---- plotting ----
    def redraw_plots(self):
        # total loss
        self.ax_total.clear()
        self.ax_total.set_title("Total Loss")
        if self.trainer.loss_history:
            self.ax_total.plot(list(self.trainer.loss_history), label="Total")
            self.ax_total.legend()

        # sub-losses (based on checkboxes)
        self.ax_sub.clear()
        self.ax_sub.set_title("Subsystem Losses")
        for k in self.trainer.sub_losses.keys():
            if self.loss_show[k].get() and self.trainer.sub_losses[k]:
                self.ax_sub.plot(list(self.trainer.sub_losses[k]), label=k)
        self.ax_sub.legend()
        self.canvas.draw()

    # ---- periodic UI update ----
    def update_gui(self):
        if self.trainer.last_loss is not None:
            self.loss_lbl.config(text=f"Loss: {self.trainer.last_loss:.4f}")
        self.step_lbl.config(text=f"Step: {self.trainer.step}")
        # refresh plots occasionally
        self.redraw_plots()
        # change status if thread finished
        if not (self.thread and self.thread.is_alive()) and not self.trainer.running:
            if "Stopping" in self.status_lbl.cget("text"):
                self.status_lbl.config(text="Status: Stopped")
        self.root.after(400, self.update_gui)

    # ---- system stats tab ----
    def update_sys(self):
        try:
            if psutil:
                self.cpu_lbl.config(text=f"CPU: {psutil.cpu_percent(interval=None):.1f} %")
                self.mem_lbl.config(text=f"Memory: {psutil.virtual_memory().percent:.1f} %")
                proc = psutil.Process(os.getpid())
                self.proc_cpu_lbl.config(text=f"Process CPU: {proc.cpu_percent(interval=None):.1f} %")
                rss_mb = proc.memory_info().rss / (1024*1024)
                self.proc_mem_lbl.config(text=f"Process RSS: {rss_mb:.1f} MB")
            else:
                # fallback (POSIX only; coarse)
                if hasattr(os, "getloadavg"):
                    load1, _, _ = os.getloadavg()
                    self.cpu_lbl.config(text=f"CPU (load1): {load1:.2f}")
                else:
                    self.cpu_lbl.config(text="CPU: n/a")
                try:
                    import resource
                    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # ru_maxrss unit varies; assume KB
                    self.mem_lbl.config(text=f"RSS (approx): {rss_kb/1024:.1f} MB")
                except Exception:
                    self.mem_lbl.config(text="Memory: n/a")
        except Exception as e:
            self.cpu_lbl.config(text=f"CPU: n/a ({e})")
            self.mem_lbl.config(text="Memory: n/a")
        self.root.after(1000, self.update_sys)

    # ---- window close ----
    def on_close(self):
        # graceful stop + save
        if self.trainer.running:
            self.trainer.stop(save=True)
        self.root.after(300, self.root.destroy)

# -----------------------------
# Importable helpers
# -----------------------------
def run_gui():
    root = tk.Tk()
    app = GaiaApp(root)
    root.mainloop()

def run_headless_forever(batch_size=16, lr=3e-4, ckpt="gaia_weights.pth", stop_flag_path="stop.flag"):
    tr = GaiaTrainer(batch_size=batch_size, lr=lr, ckpt=ckpt, stop_flag_path=stop_flag_path)
    tr.train_forever()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_gui()
