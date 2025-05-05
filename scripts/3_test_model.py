import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

model_path = "models/t5-small-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

#  Generate predictions from input texts
def generate_predictions(input_texts):
    predictions = []
    for text in tqdm(input_texts, desc="Generating predictions"):
        input_ids = tokenizer("simplify: " + text, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
        output_ids = model.generate(input_ids, max_length=128)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)
    return predictions

#  Compute token-level F1 score and plot confusion matrix
def compute_f1_and_confusion(true_texts, pred_texts, label):
    print(f"\n F1 Evaluation for {label.upper()} set")

    all_tokens = set()
    for text in true_texts + pred_texts:
        all_tokens.update(text.lower().split())
    all_tokens = sorted(list(all_tokens))

    y_true_bin = []
    y_pred_bin = []

    for ref, pred in zip(true_texts, pred_texts):
        ref_set = set(ref.lower().split())
        pred_set = set(pred.lower().split())

        y_true_bin.append([1 if token in ref_set else 0 for token in all_tokens])
        y_pred_bin.append([1 if token in pred_set else 0 for token in all_tokens])

    f1 = f1_score(np.array(y_true_bin), np.array(y_pred_bin), average="micro")
    print(f" F1 Score (micro): {f1:.4f}")

    # Confusion matrix across all tokens
    cm = confusion_matrix(np.array(y_true_bin).flatten(), np.array(y_pred_bin).flatten())
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({label.upper()} Tokens)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{label}.png")
    print(f" Saved confusion matrix as confusion_matrix_{label}.png")

    return f1

#  Evaluation runner
def evaluate_on_csv(csv_path, label):
    df = pd.read_csv(csv_path)
    if not all(col in df.columns for col in ["source", "target"]):
        raise ValueError("CSV must have 'source' and 'target' columns.")

    sources = df["source"].tolist() 
    targets = df["target"].tolist()
    preds = generate_predictions(sources)

    f1 = compute_f1_and_confusion(targets, preds, label)

    # Save predictions
    pd.DataFrame({
        "source": sources,
        "predicted": preds,
        "target": targets
    }).to_csv(f"{label}_predictions.csv", index=False)

    print(f" Saved predictions to {label}_predictions.csv")
    return f1

#  Evaluate on train and test CSVs
f1_train = evaluate_on_csv("data/cochrane_train.csv", label="train")
f1_test = evaluate_on_csv("data/cochrane_test.csv", label="test")

print("\n FINAL F1 SCORES")
print(f"Train F1: {f1_train:.4f}")
print(f"Test  F1: {f1_test:.4f}")
