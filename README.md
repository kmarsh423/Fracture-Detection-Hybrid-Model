# Fracture Detection with Hybrid CNN + Grad-CAM Triage

## Overview
This project implements a **hybrid medical imaging system for fracture detection using X-ray scans from the Stanford MURA dataset.**
The system is designed as a **high-recall screening tool** that combines:
 - A **CNN classifier** to detect the presence of fractures
 - A **Grad-CAM-based localization confidence module** to decide whether the model can confidently identify <i>where</i> a fracture is located

Rather than forcing a binary decision, the system outputs **three possible outcomes**:
 - **Negative**: no fracture detected
 - **Positive**: fracture detected by the CNN and the Grad-CAM localization confidence module
 - **Inconclusive**: fracture suspected, but localization confidence is low

This design mirrors real-world clinical workflows, where uncertain cases are deferred rather than misclassified

---

## Dataset

- **Source:** Stanford MURA (Musculoskeletal Radiographs)
- **Task:** Binary fracture classification (positive / negative)
- **Labels:** Image-level only (no bounding boxes or masks)
- **Validation split:** ~1500 positive, ~1700 negative samples

---

## Model

### CNN Backbone
- **Architecture:** ResNet-18
- **Initialization:** ImageNet pretrained
- **Output:** Single logit per image (fracture probability)

The CNN is trained with **binary cross-entropy with logits**, using class weighting to address imbalance.

---

## Hybrid Decision Logic

### Motivation
A CNN probability alone does not guarantee that the model can meaningfully localize a fracture.  
In medical imaging, **confidence without localization is unsafe**.

This system explicitly models that uncertainty.

---

### Step 1: High-Recall Screening

The classifier operates at a **low probability threshold**:

- If `p < p_low` → **Negative**
- If `p ≥ p_low` → proceed to localization analysis

This biases the system toward **high recall**, minimizing missed fractures.

---

### Step 2: Grad-CAM Localization Confidence

For suspected fractures, **Grad-CAM** is computed from the final convolutional layer.  
Two quantitative measures are extracted:

- **Entropy** — measures how focused the attention is  
- **Activated area ratio** — fraction of pixels exceeding $\alpha$

Diffuse or spatially implausible activations are treated as unreliable.

---

### Step 3: Hybrid Output

The final system output is:

$$
y =
\begin{cases}
\text{Negative}, & p < p_{\text{low}} \\
\text{Positive}, & p \ge p_{\text{low}} \;\wedge\; \text{localization confident} \\
\text{Inconclusive}, & \text{otherwise}
\end{cases}
$$

This introduces a principled **abstention mechanism**, improving safety and interpretability.

---

## Evaluation

Because the system can abstain, standard accuracy alone is insufficient.  
Reported metrics include:

- **Inconclusive rate** (abstention)
- **Coverage** (fraction of confident predictions)
- **Accuracy / Precision / Recall / F1** on confident cases
- **ROC-AUC** for the CNN classifier (threshold-independent)

Hyperparameter EDA is used to analyze **risk–coverage tradeoffs** and select a recall-leaning operating point with meaningful abstention.

---

## Project Structure
```text
data/
├── train/   # Train dataset     
├──── ...
├── valid/   # Validation dataset
├──── ...        
├── train_image_paths.csv         
├── train_labeled_studies.csv         
├── valid_image_paths.csv          
├── valid_labeled_studies.csv         
src/
├── dataset.py        # MURA dataset handling
├── model.py          # CNN backbone
├── train.py          # Training with early stopping
├── evaluate.py       # Grad-CAM Classifier + hybrid evaluation
notebooks/
├── eda.ipynb
├── hybrid_hyperparameter_eda.ipynb
├── GradCAM_visualization.ipynb
```

---


## Key Takeaways

- High recall alone is insufficient for medical imaging
- Localization confidence should influence decisions, not just explanations
- Explicit abstention leads to safer, more interpretable systems
- Hybrid ML + rule-based logic bridges black-box models and clinical reasoning

---

## Future Work

- Study-level aggregation (multi-view exams)
- Segmentation-aware localization
- Calibration of uncertainty estimates
- External dataset validation







# Fracture Detection with Hybrid CNN + Grad-CAM Triage

## Overview
This project implements a **hybrid medical imaging system for fracture detection using X-ray scans from the Standord MURA dataset.**.
The systen is designed as a **high-recall screening tool** that combines:
 - A **CNN classifier** to detect the presence of fractures
 - A **Grad-CAM-based localization confidence module** to decide whether th emodel can confidently identify <i>where</i> a fracture is located

Rather than forcing a binary decision, the system outputs **three possible outcomes**:
 - **Negative**: no fracture detected
 - **Positive**: fracture detected by the CNN and the Grad-CAM localization confidence module
 - **Inconclusive**: fracture suspected, but localization confidence is low

This design mirrors real-world clinical workflows, where uncertain cases are deferred rather than misclassified

---

## Dataset

- **Source:** Stanford MURA (Musculoskeletal Radiographs)
- **Task:** Binary fracture classification (positive / negative)
- **Labels:** Image-level only (no bounding boxes or masks)
- **Validation split:** ~1500 positive, ~1700 negative samples

---

## Model

### CNN Backbone
- **Architecture:** ResNet-18
- **Initialization:** ImageNet pretrained
- **Output:** Single logit per image (fracture probability)

The CNN is trained with **binary cross-entropy with logits**, using class weighting to address imbalance.

---

## Hybrid Decision Logic

### Motivation
A CNN probability alone does not guarantee that the model can meaningfully localize a fracture.  
In medical imaging, **confidence without localization is unsafe**.

This system explicitly models that uncertainty.

---

### Step 1: High-Recall Screening

The classifier operates at a **low probability threshold**:

- If `p < p_low` → **Negative**
- If `p ≥ p_low` → proceed to localization analysis

This biases the system toward **high recall**, minimizing missed fractures.

---

### Step 2: Grad-CAM Localization Confidence

For suspected fractures, **Grad-CAM** is computed from the final convolutional layer.  
Two quantitative measures are extracted:

- **Entropy** — measures how focused the attention is  
- **Activated area ratio** — fraction of pixels exceeding 50% of the maximum Grad-CAM response  

Diffuse or spatially implausible activations are treated as unreliable.

---

### Step 3: Hybrid Output

The final system output is:

$$
  y =
  \begin{cases}
  \text{Negative}, & p < p_{\text{low}} \\
  \text{Positive}, & p \ge p_{\text{low}} \;\wedge\; \text{localization confident} \\
  \text{Inconclusive}, & \text{otherwise}
  \end{cases}
$$
This introduces a principled **abstention mechanism**, improving safety and interpretability.

---

## Evaluation

Because the system can abstain, standard accuracy alone is insufficient.  
Reported metrics include:

- **Inconclusive rate** (abstention)
- **Coverage** (fraction of confident predictions)
- **Accuracy / Precision / Recall / F1** on confident cases
- **ROC-AUC** for the CNN classifier (threshold-independent)

Hyperparameter EDA is used to analyze **risk–coverage tradeoffs** and select a recall-leaning operating point with meaningful abstention.

---

## Conclusion

This project demonstrates a hybrid approach to fracture detection that combines a high-recall CNN classifier with Grad-CAM–based localization confidence to enable uncertainty-aware decision making. Rather than forcing a binary prediction for every X-ray image, the system explicitly allows for inconclusive outcomes when the model’s localization confidence is insufficient. This design better reflects real clinical workflows, where uncertain cases are deferred for expert review instead of being misclassified.

The figure below illustrates the risk–coverage tradeoff of the hybrid system. Each point represents a different configuration of classification and localization thresholds, evaluated in terms of coverage (the proportion of cases for which the system makes a confident decision) and accuracy on those confident cases. Color indicates the inconclusive rate, highlighting how abstention increases as coverage decreases.

![Coverage vs Accuracy Figure](./results/Coverage_vs_Accuracy.png)

As shown, increasing the inconclusive rate does not monotonically improve accuracy. While moderate abstention can remove some unreliable predictions, overly aggressive deferral also discards many correct cases, reducing overall usefulness. This behavior is expected, as Grad-CAM–based localization confidence is an imperfect proxy for correctness when only image-level labels are available.

The highlighted operating region marks a practical compromise: a moderate inconclusive rate that preserves most of the classifier’s performance while deferring genuinely uncertain predictions. In this regime, the system achieves strong recall and reliable accuracy on confident cases, without excessive abstention.

Overall, this work shows that interpretability tools such as Grad-CAM can be elevated from post-hoc explanations to active components in a decision system. By framing fracture detection as a selective prediction problem rather than a forced classification task, the hybrid approach prioritizes safety, transparency, and clinical realism—key considerations for medical machine learning applications.




