# SentinelNet

The goal of this project is to develop an AI-powered **Network Intrusion Detection System (NIDS)** capable of identifying malicious network traffic and cyber-attacks in real time. By leveraging machine learning techniques, the system classifies traffic as normal or suspicious based on historical data.

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

The project uses the **NSL-KDD dataset**, a widely used benchmark for network intrusion detection.  

- **Training dataset:** `KDDTrain+.txt`  
- Each record describes a **network connection** with multiple features  
- Labels indicate whether the traffic is **normal** or an **attack** (various categories)  

For more details, see: [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)

## ⚙️ Installation

1. **Clone the repository**
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

