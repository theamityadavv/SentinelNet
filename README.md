# 🛡️ SentinelNet

**SentinelNet** is an AI-powered Network Intrusion Detection System (NIDS) capable of identifying malicious network traffic and cyber-attacks in real time. By leveraging machine learning techniques, the system classifies network traffic as normal or suspicious based on historical data, so organizations can detect intrusions faster and more accurately than with traditional systems.

---

## 📂 Project Structure

````text
SentinelNet/
├─ data/
│  └─ NSL-KDD/
│     └─ KDDTrain+.txt
├─ analysis/
│  └─ nsl_kdd_analysis.py
├─ notebooks/
│  └─ load_and_explore.py
├─ docs/
│  └─ data_overview.md
└─ README.md

---

## 🐍 Setup & Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SpringBoardMentor193s/SentinelNet.git
cd SentinelNet
````

2. **Create and activate a virtual environment:**

```bash
python -m venv .venv
# Windows PowerShell
& .venv/Scripts/Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

* **Run analysis script:**

```bash
python analysis/nsl_kdd_analysis.py
```

* **Run Jupyter notebook:**

```bash
jupyter notebook notebooks/load_and_explore.py
```

---

## 📄 Dataset

The project uses the **NSL-KDD dataset**, which includes features describing network connections and labels for attack types or normal traffic.

* **Training dataset:** `https://github.com/SpringBoardMentor193s/SentinelNet/blob/amityadav/data/NSL-KDD/KDDTrain%2B.txt`
* Explore dataset via Pandas for statistics and visualizations.

##  CICIDS2017 Dataset

Since the dataset is too large for GitHub, you can download and extract it automatically:

```bash
pip install gdown
python download_data.py


---

## 📖 Documentation

Refer to [Data Overview](https://github.com/SpringBoardMentor193s/SentinelNet/blob/10bb40432b8b25131207bacf99b0d9a88d76481c/docs/data_overview.md) for details on dataset sources, schema, and summary statistics.

---
👥 Team Members
<div style="overflow-x:auto;"> <table cellpadding="5" cellspacing="0"> <tr><th>Name</th><th>Email</th></tr> <tr><td>Amit Yadav</td><td><a href="mailto:2k23.cs2314011@gmail.com">2k23.cs2314011@gmail.com</a></td></tr> <tr><td>Upasana Prabhakar</td><td><a href="mailto:upasanaprabhakar35@gmail.com">upasanaprabhakar35@gmail.com</a></td></tr> <tr><td>Bhavana Thota</td><td><a href="mailto:bhavanathota2006@gmail.com">bhavanathota2006@gmail.com</a></td></tr> <tr><td>Saravanan S</td><td><a href="mailto:itssaravanan03@gmail.com">itssaravanan03@gmail.com</a></td></tr> <tr><td>Shreyanshi Srivastava</td><td><a href="mailto:shreyanshisrivastava19@gmail.com">shreyanshisrivastava19@gmail.com</a></td></tr> <!-- Add remaining members here --> </table> </div>
