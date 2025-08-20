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


## 📊 Dataset  

<p align="center">
The project uses the <b>NSL-KDD dataset</b>, a widely used benchmark for <b>network intrusion detection</b>.  
</p>

- 📘 **Training dataset:** `KDDTrain+.txt`  
- 📡 Each record describes a <b>network connection</b> with multiple features  
- ⚔️ Labels indicate whether the traffic is <b>normal</b> or an <b>attack</b> (various categories)  

<p align="center">
🔗 <a href="https://www.kaggle.com/datasets/hassan06/nslkdd" target="_blank"><b>NSL-KDD Dataset (Kaggle)</b></a>
</p>


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

