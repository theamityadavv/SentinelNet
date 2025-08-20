<h1 align="center">🛡️ SentinelNet</h1>

<p align="center">
An AI-powered <b>Network Intrusion Detection System (NIDS)</b> that detects malicious traffic and cyber-attacks in real time using Machine Learning.
</p>

---

## 📂 Project Structure

```text
SentinelNet/
├─ data/
│  └─ NSL-KDD/
│     └─ KDDTrain+.txt
├─ analysis/
│  └─ nsl_kdd_analysis.py
├─ notebooks/
│  └─ load_and_explore.py
└─ README.md

---
## 🐍 2. Create and Activate a Virtual Environment  

Set up a dedicated **Python virtual environment** to keep dependencies isolated and organized.  

```bash
# Create a new virtual environment
python -m venv .venv

# ▶️ Activate on Windows PowerShell
& .venv/Scripts/Activate.ps1

# 🐧 Activate on Linux/Mac
source .venv/bin/activate


## ⚙️ Installation  

Follow these steps to set up the project on your local machine:  

---

### 🔹 1. Clone the repository  
```bash
git clone https://github.com/SpringBoardMentor193s/SentinelNet.git
cd SentinelNet


### 2. Create and activate a virtual environment

python -m venv .venv
# Windows PowerShell
& .venv/Scripts/Activate.ps1
# Linux/Mac
source .venv/bin/activate

### 3. Install dependencies

Install all required Python libraries using:

```bash
pip install -r requirements.txt


## 🚀 Usage

- **Run the analysis script**
  ```bash
  python analysis/nsl_kdd_analysis.py

- **Explore via Jupyter Notebook**
  ```bash
  jupyter notebook notebooks/load_and_explore.py

