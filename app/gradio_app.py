import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load fine-tuned model and tokenizer
model_path = "models/t5-small-finetuned"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Long medical texts with clear labels
example_texts = {
    "🫁 Chest Physiotherapy in Cystic Fibrosis":
    """We included six Cochrane Reviews, one of which compared any type of chest physiotherapy with no chest physiotherapy or coughing alone and the remaining five reviews included head-to-head comparisons of different airway clearance techniques. All the reviews were considered to have a low risk of bias. However, the individual trials included in the reviews often did not report sufficient information to adequately assess risk of bias. Many trials did not sufficiently report on outcome measures and had a high risk of reporting bias. We are unable to draw definitive conclusions for comparisons of airway clearance techniques in terms of FEV1, except for reporting no difference between PEP therapy and oscillating devices after six months of treatment, mean difference -1.43% predicted (95% confidence interval -5.72 to 2.87); the quality of the body of evidence was graded as moderate. The quality of the body of evidence comparing different airway clearance techniques for other outcomes was either low or very low. There is little evidence to support the use of one airway clearance technique over another. People with cystic fibrosis should choose the airway clearance technique that best meets their needs, after considering comfort, convenience, flexibility, practicality, cost, or some other factor. More long-term, high-quality randomised controlled trials comparing airway clearance techniques among people with cystic fibrosis are needed.""",

    "🧪 Parenteral Calcium Use in ICU":
    """There are no identifiable studies that have evaluated the association between parenteral calcium supplementation in critically ill ICU patients and the following outcomes: mortality, multiple organ dysfunction, ICU and hospital length of stay, costs, and complications of calcium administration. Serum ionized calcium concentration was reported in 5 studies (12 trial arms, 159 participants). These trials showed a small but significant increase in serum ionized calcium concentration after calcium administration. These trials showed considerable statistical heterogeneity and differed extensively in the population studied (adult versus neonate), the indication (hypocalcemia versus prophylaxis) and threshold of hypocalcemia for which parenteral calcium was administered, and the timing of subsequent measurement of serum ionized calcium concentration to the extent that we consider a pooled estimate almost inappropriate. There is no clear evidence that parenteral calcium supplementation impacts the outcome of critically ill patients.""",

    "🦵 CPM Therapy for Preventing VTE after Knee Surgery":
    """Eleven RCTs involving 808 participants met the inclusion criteria. The methodological quality of the included studies was variable and most of the predefined outcomes were reported by only one or two studies, therefore the quality of the evidence was low. Five studies with a total of 405 patients reported the incidence of DVT. In the CPM group (205 patients) 36 developed DVT (18%) compared to 29 (15%) in the control group (200 patients). The results of the meta-analysis showed no evidence that CPM had any effect on preventing VTE after TKA (RR 1.22, 95% CI 0.84 to 1.79). One trial (150 participants) did not find PE in any of the patients during hospitalisation or in the subsequent three months. PE was not reported in the other included studies. None of the trials reported deaths among the included participants. There is not enough evidence from the available RCTs to conclude that CPM reduces VTE after TKA. We cannot assess the effect of CPM on mortality because no such events occurred amongst the participants of these trials. The quality of the evidence was low. The results are supported by only a small number of studies, most of which are of low to moderate quality."""
}

# Simplification function
def simplify_text(text):
    input_text = "simplify: " + text
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Medical Text Simplifier")
    gr.Markdown("Use the dropdown to choose a complex medical review paragraph and simplify it into plain language.")

    dropdown = gr.Dropdown(label="📄 Choose Example", choices=list(example_texts.keys()))
    input_box = gr.Textbox(label="📝 Input Text", lines=14, placeholder="Paste your medical text here...")
    simplify_btn = gr.Button("🔍 Simplify")
    output_box = gr.Textbox(label="✅ Simplified Output", lines=6)

    # Events
    dropdown.change(fn=lambda choice: example_texts.get(choice, ""), inputs=dropdown, outputs=input_box)
    simplify_btn.click(fn=simplify_text, inputs=input_box, outputs=output_box)

demo.launch()
