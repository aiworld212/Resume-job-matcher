# 🎯 Resume–Job Matching System

> An end-to-end NLP project that matches resumes to job descriptions using **Sentence Transformers** and **Cosine Similarity** — deployed as an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![NLP](https://img.shields.io/badge/NLP-Sentence--Transformers-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project solves a real-world problem — **matching the right resume to the right job** — using semantic similarity instead of simple keyword matching.

Given a resume, the system finds the best matching jobs from 24 categories by understanding the **meaning** of the text, not just counting keywords.

---

## 🖥️ Demo

| Match Results | Skill Gap Analysis |
|---|---|
| Top jobs ranked by match score | Shows matched vs missing skills |

---

## 🧠 How It Works

```
Resume Text
     ↓
Text Cleaning  →  remove URLs, emails, HTML, special characters
     ↓
Sentence Transformer (all-MiniLM-L6-v2)
     ↓
384-dimensional Embedding Vector
     ↓
Cosine Similarity  ←→  Job Description Embeddings
     ↓
Ranked Results with Match Score (0–100%)
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Top-1 Accuracy | ~75% |
| Top-3 Accuracy | ~90% |
| Top-5 Accuracy | ~95% |
| Total Resumes | 2,484 |
| Job Categories | 24 |

---

## 🗂️ Project Structure

```
resume-job-matcher/
├── app.py                          ← Streamlit web app
├── Resume_Job_Matcher.ipynb        ← Full ML pipeline notebook
├── README.md
├── .gitignore
└── Resume.csv             ← Dataset (download from Kaggle)
   
```

---

## 📦 Dataset

This project uses the **Resume Dataset** from Kaggle.

👉 Download here: [https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

After downloading, place the file at:
```
data/raw/Resume.csv
```

| Column | Description |
|---|---|
| ID | Unique resume ID |
| Resume_str | Plain text of the resume |
| Resume_html | HTML version (not used) |
| Category | Job category label |

**24 Categories:** Information Technology, Finance, HR, Healthcare, Engineering, Sales, Accountant, Chef, Advocate, Fitness, Aviation, Banking, Construction, Public Relations, Designer, Arts, Teacher, Apparel, Digital Media, Agriculture, Automobile, BPO, Consultant, Business Development

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/aiworld212/resume-job-matcher.git
cd resume-job-matcher
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download `Resume.csv` from Kaggle and place it at `Resume.csv`

### 5. Run the notebook
Open `Resume_Job_Matcher.ipynb` and run all cells top to bottom

### 6. Launch the web app
```bash
python -m streamlit run app.py
```

App opens at **http://localhost:8501** 🎉

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| `sentence-transformers` | Generate semantic embeddings |
| `scikit-learn` | Cosine similarity, t-SNE, evaluation |
| `pandas` & `numpy` | Data processing |
| `matplotlib` & `seaborn` | Visualizations |
| `plotly` | Interactive charts |
| `streamlit` | Web application |
| `nltk` | Text preprocessing |

---

## 📈 Notebook Pipeline

| Step | File | Description |
|---|---|---|
| 1 | Notebook Step 1 | Exploratory Data Analysis |
| 2 | Notebook Step 2 | Text cleaning + job descriptions |
| 3 | Notebook Step 3 | Generate sentence embeddings |
| 4 | Notebook Step 4 | Match resumes + evaluate accuracy |
| 5 | app.py | Interactive Streamlit web app |

---

## 💡 Key Features

- ✅ Semantic similarity — understands meaning, not just keywords
- ✅ 24 job category matching
- ✅ Skill gap analysis — shows matched vs missing skills
- ✅ Interactive Streamlit UI with charts
- ✅ Support for PDF, DOCX, and TXT resume upload
- ✅ t-SNE embedding visualization
- ✅ Top-1 / Top-3 / Top-5 accuracy evaluation

---

## 👤 Author

**Your Name**
- GitHub: [@aiworld212](https://github.com/aiworld212)

---

## 📄 License

This project is licensed under the MIT License.
