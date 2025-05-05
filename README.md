# 🧠 Medical Text Simplifier using T5-small

A lightweight Gradio-based web app that simplifies complex medical review paragraphs into plain language using a fine-tuned T5 model. The system is tailored for researchers, clinicians, and patients who wish to better understand medical content.

---

## 📌 Project Summary

Medical literature is often filled with technical jargon and statistical data that can be difficult for non-experts to grasp. This project utilizes a fine-tuned version of the T5 transformer model to simplify medical review texts into more accessible language, improving readability and understanding.

---

## ✅ Features

- Text input only — no image/OCR support
- Fine-tuned `t5-small` model on Cochrane dataset
- Gradio-based web interface for easy use
- Outputs simplified medical content in real-time
- Evaluation via prediction CSV and confusion matrix

---

## 🧰 Tech Stack

| Component        | Technology              |
|------------------|--------------------------|
| Model            | T5-small (fine-tuned)    |
| Frontend         | Gradio                   |
| Data             | Cochrane Review Dataset  |
| Backend Scripting| Python (Transformers, pandas) |

---

## 📁 Project Structure

```
Medical_Instructions_Summary/
├── app/
│   └── gradio_app.py            # Gradio web app
├── data/
│   ├── cochrane_train.csv
│   ├── cochrane_test.csv
│   └── cochrane_val.csv
├── models/
│   └── t5-small-finetuned/      # Saved model checkpoints and config
│       ├── checkpoint-XXXX/
│       ├── config.json
│       ├── tokenizer_config.json
│       └── model.safetensors
├── scripts/
│   ├── 1_preprocess.py
│   ├── 2_train.py
│   ├── 3_test_model.py
├── test_predictions.csv
├── train_predictions.csv
├── confusion_matrix_test.png
├── confusion_matrix_train.png
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/Medical_Instructions_Summary.git
cd Medical_Instructions_Summary
```

### 2. Run the app

```bash
cd app
python gradio_app.py
```

---

## 🧪 Example

**Input Text**:
```
Eleven RCTs involving 808 participants met the inclusion criteria...
```

**Simplified Output**:
```
We included 11 randomised controlled trials (RCTs) involving 808 participants in this review...
```

---

## 📊 Evaluation

- `test_predictions.csv` and `train_predictions.csv` contain model outputs
- `confusion_matrix_train.png` and `confusion_matrix_test.png` visualize classification quality

---

## 👥 Contributors

- Gurman Singh Marahar  
- Mamta Rani

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- Cochrane Review Dataset
- Hugging Face Transformers
- Gradio
