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


📊 Dataset

The project uses the NSL-KDD dataset, a widely used benchmark for network intrusion detection.

Training dataset: KDDTrain+.txt

Each record describes a network connection with multiple features.

Labels indicate whether the traffic is normal or an attack (various categories).

For more details, see: NSL-KDD Dataset

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SpringBoardMentor193s/SentinelNet.git
cd SentinelNet

2. Create and activate a virtual environment:

python -m venv .venv
# Windows PowerShell
& .venv/Scripts/Activate.ps1
# Linux/Mac
source .venv/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Usage

Analysis script: analysis/nsl_kdd_analysis.py

Jupyter notebook: notebooks/load_and_explore.py

Run the notebook or script to explore the NSL-KDD dataset, analyze attack categories, and perform basic data preprocessing.
