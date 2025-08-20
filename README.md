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


<h3 align="center">🔹 2. Create and Activate a Virtual Environment</h3>

<p align="center">
To keep dependencies isolated, create a <b>virtual environment</b> and activate it based on your operating system.
</p>

<hr/>

<h4>💻 Create Environment</h4>
<pre><code>python -m venv .venv</code></pre>

<hr/>

<h4>🪟 Windows PowerShell</h4>
<pre><code>&amp; .venv/Scripts/Activate.ps1</code></pre>

<hr/>

<h4>🐧 Linux / Mac</h4>
<pre><code>source .venv/bin/activate</code></pre>


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

