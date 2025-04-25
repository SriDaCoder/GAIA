import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import random
import time
import matplotlib.pyplot as plt

# AI Names
ai_names = ["HADES", "Προμηθεύς", "DEMETER", "POSEIDON", "AETHER"]

# AI Performance Storage (no JSON)
performance = {ai: {"accuracy": 0, "response_time": 0, "success_rate": 0} for ai in ai_names}

# AI Decision-Making Logic (GAIA)
def ai_coordinator():
    try:
        # Simulate AI performance monitoring
        while True:
            update_performance()
            time.sleep(5)  # Update every 5 seconds
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred. Technical Details:\n{e}")
        # Log error to a file (for debugging)
        print(f"Error: {e}")

def update_performance():
    for ai in ai_names:
        # Adjust AI performance dynamically (simulated)
        performance[ai]["accuracy"] = round(random.uniform(70, 99), 2)
        performance[ai]["response_time"] = round(random.uniform(100, 500), 2)
        performance[ai]["success_rate"] = round(random.uniform(80, 99), 2)

        # AI intervention logic (simulating GAIA-like control)
        if performance[ai]["accuracy"] < 75:
            print(f"GAIA: Boosting {ai}'s accuracy...")
            performance[ai]["accuracy"] += random.uniform(5, 10)

        if performance[ai]["response_time"] > 400:
            print(f"GAIA: Optimizing {ai}'s response time...")
            performance[ai]["response_time"] -= random.uniform(20, 50)

        if performance[ai]["success_rate"] < 85:
            print(f"GAIA: Enhancing {ai}'s success rate...")
            performance[ai]["success_rate"] += random.uniform(3, 7)

# Function to run AI scripts
def run_ai(ai_name):
    subprocess.Popen(["python", f"{ai_name}\main-test.py", ai_name])

# Initializing the tkinter window (root)
root = tk.Tk()

# tkinter UI
root.title("GAIA AI Coordinator")
root.geometry("800x600")

tk.Button(root, text="Start All AIs", command=lambda: [run_ai(ai) for ai in ai_names]).pack(pady=10)
tk.Button(root, text="Stop All AIs", command=lambda: [subprocess.Popen(["pkill", "-f", ai]) for ai in ai_names]).pack(pady=10)

tk.Button(root, text="Optimize the AIs", command=lambda: threading.Thread(target=ai_coordinator).start()).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

# Graphs
tk.subheader("AI Performance Metrics")
fig, ax = plt.subplots(3, 1, figsize=(7, 12))

# Accuracy Graph
ax[0].bar(ai_names, [performance[ai]["accuracy"] for ai in ai_names], color='blue')
ax[0].set_title("Accuracy (%)")
ax[0].set_ylim(0, 100)

# Response Time Graph
ax[1].bar(ai_names, [performance[ai]["response_time"] for ai in ai_names], color='red')
ax[1].set_title("Response Time (ms)")

# Success Rate Graph
ax[2].bar(ai_names, [performance[ai]["success_rate"] for ai in ai_names], color='green')
ax[2].set_title("Success Rate (%)")
ax[2].set_ylim(0, 100)

tk.pyplot(fig)