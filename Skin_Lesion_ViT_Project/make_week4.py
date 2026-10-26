"""Week 4 deliverables (Research Framing and Proposal Development): report DOCX + PDF + PPTX.
Follows the workshop brief: no methodology/experiments. Run: python make_week4.py"""
import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_documents as md

TITLE = "Data-Efficient Vision Transformers for Skin Lesion Classification under Limited and Imbalanced Training Data"
md.TITLE = TITLE
md.SUBTITLE = "Week 4 Report: Research Framing and Proposal Development"
md.EXTRA = ["Instructor: Dr. Milan Dordevic"]
md.SHORT = "Week 4 | Research Framing and Proposal Development"

CLASSES = [("akiec", "Actinic keratoses / intraepithelial carcinoma", 327), ("bcc", "Basal cell carcinoma", 514),
           ("bkl", "Benign keratosis-like lesions", 1099), ("df", "Dermatofibroma", 115),
           ("mel", "Melanoma", 1113), ("nv", "Melanocytic nevi", 6705), ("vasc", "Vascular lesions", 142)]

REFS = [
    "Ansari, F., Chakraborti, T., & Das, S. (2024). Algorithmic fairness in lesion classification by mitigating class imbalance and skin tone bias. Medical Image Computing and Computer Assisted Intervention (MICCAI 2024).",
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR 2021).",
    "He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).",
    "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016).",
    "Liu, Y., Sangineto, E., Bi, W., Sebe, N., Lepri, B., & De Nadai, M. (2021). Efficient training of visual transformers with small datasets. Advances in Neural Information Processing Systems (NeurIPS 2021).",
    "Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. IEEE/CVF International Conference on Computer Vision (ICCV 2021).",
    "Rahmat, I. H., Gan, J. Q., & Raza, H. (2024). Enhancing skin lesion classification: A self-attention fusion approach with Vision Transformer. Medical Image Understanding and Analysis (MIUA 2024).",
    "Shajimon, G. M., Ufumaka, I., & Raza, H. (2023). An improved vision-transformer network for skin cancer classification. IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2023).",
    "Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jegou, H. (2021). Training data-efficient image transformers & distillation through attention. International Conference on Machine Learning (ICML 2021).",
    "Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161",
    "Xiao, J., Bai, Y., Yuille, A., & Zhou, Z. (2023). Delving into masked autoencoders for multi-label thorax disease classification. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023).",
    "Xie, Z., Zhang, Z., Cao, Y., et al. (2022). SimMIM: A simple framework for masked image modeling. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).",
]

TOPIC = ("Skin lesion classification from dermoscopic images is an active area of computer vision with the potential to "
 "support early detection of skin cancer. Automated models learn visual patterns that separate diagnoses such as melanoma, "
 "basal cell carcinoma and benign nevi, but their reliability depends on the quantity and balance of the labelled data "
 "used for training. Expert-labelled medical images are expensive to collect, and public datasets contain many examples "
 "of common diagnoses and few of rare ones. Two linked challenges follow: limited labelled data and class imbalance. A "
 "model can reach a high overall accuracy while failing on under-represented, clinically important classes.\n"
 "Vision Transformers (ViTs) split an image into patches and use self-attention to relate them (Dosovitskiy et al., 2021). "
 "They are powerful at scale but have weaker built-in image priors than convolutional networks and need considerable data "
 "(Liu et al., 2021). Self-supervised pre-training, data-efficient auxiliary tasks and imbalance-aware learning have been "
 "proposed as remedies, yet evidence from natural images or other medical modalities does not automatically transfer to "
 "dermoscopy. This research examines how reliably Vision Transformers classify skin lesions when labelled data and "
 "minority-class examples are both limited, emphasising class-wise performance rather than accuracy alone.")

PAPERS = [
 ("Paper 1: Efficient Training of Visual Transformers with Small Datasets (Liu et al., 2021)", [
  "Contribution and findings. Liu et al. (2021) investigated why Vision Transformers can behave very differently from convolutional networks when trained on small datasets, even when their results on large benchmarks are similar. They proposed an approach intended to improve learning when labelled examples are scarce and reported improvements across several transformer architectures and datasets without a substantial increase in computational cost.",
  "Methodology. The authors added an auxiliary self-supervised task, Dense Relative Localization (DRLoc), to ordinary supervised training. DRLoc encourages the network to learn spatial relationships between different image regions, which gives the transformer an additional learning signal that does not require extra labels.",
  "Limitations. The study addresses general computer vision rather than medical images, so it does not establish that the benefits carry over to dermoscopy. It also does not directly tackle the shortage of examples for rare or malignant lesions. Its value here is as evidence that small-data behaviour of ViTs is a distinct problem that self-supervision can help address."]),
 ("Paper 2: Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification (Xiao et al., 2023)", [
  "Contribution and findings. Xiao et al. (2023) studied medical-image-specific pre-training as a way to improve Vision Transformers when labelled medical data are limited, focusing on multi-label thorax disease classification. Their results indicated that pre-training on medical images allows a ViT to reach results competitive with established convolutional networks; their ViT-S model was competitive with DenseNet-121 on three public chest X-ray datasets.",
  "Methodology. The authors pre-trained ViTs with Masked Autoencoders (MAE) on 266,340 unlabelled chest X-ray images. During pre-training, portions of each image are masked and the model learns to reconstruct the missing pixels, which teaches useful visual representations without labels. The pre-trained models were then evaluated on the three public datasets.",
  "Limitations. The work concerns chest X-rays and thoracic diseases, whose image characteristics differ from dermoscopic images, so the findings cannot be assumed to transfer to skin lesion classification. The approach also relies on a large collection of unlabelled medical images. It supports investigating self-supervised pre-training while underlining the need for dermatology-specific evaluation."]),
 ("Paper 3: Algorithmic Fairness in Lesion Classification by Mitigating Class Imbalance and Skin Tone Bias (Ansari et al., 2024)", [
  "Contribution and findings. Ansari et al. (2024) addressed two concerns in skin lesion classification: unequal representation of lesion categories and potential bias linked to skin tone. Their work highlights that a model trained mainly on common lesion types or on limited skin-tone representation may be less reliable for under-represented groups, which makes class-wise evaluation essential.",
  "Methodology. The authors proposed a MixUp-based augmentation approach that creates additional training examples for minority classes, together with an adaptive sampler that changes which examples are selected according to model performance on a meta-set. The approach was evaluated on the ISIC 2018 and ASAN skin lesion datasets.",
  "Limitations. The review notes for this paper raise questions about how completely the method addresses skin-tone bias and how novel it is relative to existing augmentation and sampling techniques. The evaluation used a limited set of model configurations, and the paper suggests integrating contrastive learning in future work. Class imbalance and skin-tone bias are related but distinct problems, and data efficiency across transformer configurations was not examined."]),
 ("Paper 4: Enhancing Skin Lesion Classification: A Self-Attention Fusion Approach with Vision Transformer (Rahmat et al., 2024)", [
  "Contribution and findings. Rahmat et al. (2024) proposed a Vision Transformer framework that combines several attention mechanisms for skin lesion classification. Unlike work evaluated only on natural images or other medical modalities, this study examines a transformer-based approach directly on a dermatology dataset. The abstract reports an F1-score improvement of more than 12% compared with the authors' baseline.",
  "Methodology. The framework combines scaled dot-product, multiplicative and additive attention. The researchers explored different attention combinations, loss functions and fusion strategies using the ISIC 2017 dataset.",
  "Limitations. The reported experiments focus on ISIC 2017, so they do not alone establish generalisation to other datasets or clinical settings. The available evidence also does not show how robust the method remains when training data are substantially reduced or class imbalance becomes more severe. These are limitations of the evidence rather than weaknesses acknowledged by the authors. The study clarifies attention design but leaves the combination of limited data and severe imbalance open."]),
 ("Paper 5: An Improved Vision-Transformer Network for Skin Cancer Classification (Shajimon et al., 2023)", [
  "Contribution and findings. Shajimon et al. (2023) developed an improved Vision Transformer for skin cancer classification and compared it with other approaches, considering how transformer architecture and training choices can be adapted to dermatology images. The authors reported competitive performance compared with CNN-based methods and a soft-attention model.",
  "Methodology. Using the ISIC 2017 dataset, the authors modified a ViT-B16-based model with additional layers and regularisation. Images were resized to 224 × 224 pixels and augmentation was used to increase the number of training examples. They also applied focal loss, which places more emphasis on difficult examples, and five-fold cross-validation.",
  "Limitations. The authors note challenges with visually similar lesion categories, variation within the same category and classification for darker skin tones, and recommend larger datasets. The experiments focus on binary classification, so they do not establish performance across a broader multi-class diagnostic setting. The paper is a useful reference for transformer baselines, augmentation and imbalance-aware loss functions."]),
]

SYNTH = [
 "The five papers approach the research problem from complementary directions. Liu et al. (2021) and Xiao et al. (2023) both rely on self-supervision to make transformers less dependent on labels, but the former uses an auxiliary spatial task on general images while the latter uses masked reconstruction on a large unlabelled chest X-ray collection. Both therefore support self-supervised learning as a promising direction, yet neither studies skin lesions.",
 "Ansari et al. (2024), Rahmat et al. (2024) and Shajimon et al. (2023) work directly on dermoscopic data. They differ in what they target: Ansari et al. focus on imbalance and skin-tone bias through augmentation and sampling, Rahmat et al. on attention design, and Shajimon et al. on a modified ViT with augmentation and focal loss. Their datasets (ISIC 2017, ISIC 2018, ASAN) and tasks (binary and multi-class) also differ, so reported scores are not directly comparable.",
 "A common pattern is that improvements are reported at the full available training size. What the papers do not show, taken together, is how a transformer's class-wise reliability changes as labelled data are progressively reduced in a multi-class dermatology setting. A defensible gap must also acknowledge that parts of the problem have been studied; novelty therefore lies in the specific intersection rather than in any single component."]

GAP = [
 "The reviewed literature offers useful techniques for small-data transformer training (Liu et al., 2021), medical-image pre-training (Xiao et al., 2023), imbalance-aware learning (Ansari et al., 2024), attention design (Rahmat et al., 2024) and transformer adaptation with augmentation and focal loss (Shajimon et al., 2023). Read together, however, these studies leave the following specific gap.",
 "Missing: a class-wise evaluation of how self-supervised, data-efficient Vision Transformers perform on multi-class dermoscopic images when labelled data and minority-class examples are limited at the same time.",
 "The connection to the reviewed papers is direct. Liu et al. (2021) examine small data on general images, and Xiao et al. (2023) examine pre-training on chest X-rays, so neither tests dermoscopy. Ansari et al. (2024) treat imbalance in skin lesions but do not settle how transformer configurations behave under reduced data. Rahmat et al. (2024) and Shajimon et al. (2023) classify skin lesions with transformers but report results at full training size, and Shajimon et al. use a binary task. None of the five reports class-wise performance as the labelled set shrinks.",
 "The research need is justified by the clinical setting. Rare classes such as melanoma and vascular lesions are among the hardest to learn because they have few examples, yet errors on them are the most consequential. Overall accuracy can hide such failures, so macro-F1 and per-class recall are appropriate measures for the proposed work. This gap is derived from the five reviewed papers and is not a claim that no other study has addressed it; a targeted literature search should confirm it before a later study is planned."]

PROBLEM = ("It remains unclear how reliably Vision Transformer classifiers recognise minority diagnostic classes of skin "
 "lesions when both the labelled training data and the minority-class examples are limited. This research addresses the gap by "
 "examining whether masked-image self-supervised pre-training improves class-wise reliability, measured by macro-F1 and "
 "per-class recall, compared with a conventionally trained Vision Transformer.")

HYP = [
 "H1: Under a limited labelled-data setting, a Vision Transformer initialised with masked-image self-supervised pre-training achieves a higher macro-F1 score on a multi-class, imbalanced skin lesion dataset than the same Vision Transformer trained conventionally from scratch.",
 "H0: Under the same limited labelled-data setting, the pre-trained Vision Transformer does not achieve a higher macro-F1 score than the conventionally trained Vision Transformer.",
 "The hypothesis is testable because both models can be compared on the same held-out test set using one numerical measure. Macro-F1 gives each diagnostic class equal weight, so it is more informative than accuracy when class frequencies differ; per-class recall is a supporting measure that shows whether rare diagnoses are being missed (Ansari et al., 2024). The hypothesis is a prediction to be tested, not a reported finding, and the exact definition of a limited setting will be fixed before any future evaluation."]


def wc(t):
    return len(re.findall(r"\b\w[\w'’-]*\b", t))


def build(figdir):
    B = []
    h1 = lambda t: B.append(("h1", t)); h2 = lambda t: B.append(("h2", t)); p = lambda t: B.append(("p", t))
    h1("1. Introduction and Topic Description")
    for para in TOPIC.split("\n"):
        p(para)
    h1("2. Literature Review")
    p("The review covers exactly five base papers on small-data transformer learning, medical-image pre-training, class "
      "imbalance and fairness, attention fusion and improved ViT-based skin cancer classification. Each summary states the "
      "contribution, a high-level view of the method, and the limitations or unresolved questions. Only the described "
      "high-level methods are covered; implementation is outside the scope of this report.")
    for title, paras in PAPERS:
        h2(title)
        for t in paras:
            p(t)
    h2("2.6 Cross-paper Synthesis")
    for t in SYNTH:
        p(t)
    B.append(("table", "Table 1. Comparison of the five reviewed papers.",
              ["Paper", "Focus", "Data", "Key technique", "Open question for this topic"], [
        ["Liu et al. (2021)", "Small-data ViT training", "General images", "DRLoc auxiliary task", "Not dermoscopy"],
        ["Xiao et al. (2023)", "Medical pre-training", "Chest X-rays", "Masked autoencoders", "Other modality; large unlabelled set"],
        ["Ansari et al. (2024)", "Imbalance and skin tone", "ISIC 2018, ASAN", "MixUp, adaptive sampler", "Data efficiency not studied"],
        ["Rahmat et al. (2024)", "Attention design", "ISIC 2017", "Attention fusion", "Reduced data not tested"],
        ["Shajimon et al. (2023)", "Improved ViT", "ISIC 2017", "Focal loss, augmentation", "Binary task only"]]))
    h1("3. Research Gap Analysis")
    for t in GAP:
        B.append(("note", t) if t.startswith("Missing:") else ("p", t))
    h1("4. Problem Statement")
    p(PROBLEM)
    p("The statement is focused on one task and two constraints identified in the literature. It is researchable because "
      "the suitability of candidate models can be assessed with class-sensitive measures without assuming the outcome in advance.")
    h1("5. Dataset Selection and Justification")
    h2("5.1 Proposed dataset")
    p("HAM10000 (Human Against Machine with 10,000 training images) is proposed as the dataset (Tschandl et al., 2018). It contains "
      "10,015 dermatoscopic images of pigmented skin lesions in seven diagnostic categories and is a widely used benchmark "
      "for skin lesion classification. Table 2 and Figure 1 show the class distribution.")
    tot = sum(c[2] for c in CLASSES)
    B.append(("table", "Table 2. HAM10000 diagnostic classes (Tschandl et al., 2018).", ["Class", "Diagnosis", "Images", "Share"],
              [[a, b, f"{c:,}", f"{100*c/tot:.1f}%"] for a, b, c in CLASSES]))
    B.append(("fig", str(figdir / "fig_ham_counts.png"), "Figure 1. Number of HAM10000 images per class (log scale).", 4.6))
    h2("5.2 Appropriateness")
    p("The image modality matches the target task, and the seven classes support class-wise analysis. Melanocytic nevi make up "
      "about two thirds of the images, while dermatofibroma and vascular lesions together make up under three percent, so "
      "the dataset reproduces the imbalance that motivates the research question. With roughly ten thousand images it is large "
      "enough to define progressively smaller labelled subsets yet small enough to be used without specialised infrastructure. "
      "The dataset is publicly available from its official source, which makes the work reproducible.")
    h2("5.3 Limitations")
    p("HAM10000 is a curated research dataset and does not represent all skin lesions, skin tones or clinical settings, so "
      "performance on it alone would not demonstrate clinical readiness. Several images can show the same lesion, which "
      "requires grouping by lesion in any future split to avoid leakage. The reviewed papers use ISIC 2017, ISIC 2018 and ASAN, "
      "so results will not be directly comparable with them. The official documentation should be checked for exact counts, "
      "provenance and usage terms before use.")
    h1("6. Research Hypothesis")
    for t in HYP:
        p(t)
    h1("7. Model Selection and Justification")
    h2("7.1 Selected model")
    p("The selected model is a small Vision Transformer, ViT-S/16 (patch size 16, about 22 million parameters), initialised by "
      "masked image modelling pre-training in the style of Masked Autoencoders (He et al., 2022) and SimMIM (Xie et al., 2022), "
      "and compared with the same architecture trained from scratch. Dosovitskiy et al. (2021) introduced the ViT architecture "
      "that this choice builds on.")
    h2("7.2 Technical capabilities relevant to the problem")
    p("A ViT represents an image as a sequence of patches and uses self-attention to model relationships between distant regions, "
      "which suits lesions whose diagnostic cues include global asymmetry, colour distribution and border structure. "
      "Masked image modelling hides part of the image and reconstructs it, so it learns from unlabelled images and does not "
      "depend on the scarce labels. Xiao et al. (2023) showed that this kind of pre-training can make a small ViT competitive "
      "with a CNN in a medical domain, which provides a reason to test it for dermoscopy.")
    h2("7.3 Why this model over alternatives")
    B.append(("table", "Table 3. Alternatives considered.", ["Alternative", "Reason it was not chosen as the primary model"], [
        ["ResNet-style CNN (He et al., 2016)", "Strong image priors, but the research question concerns transformer suitability; kept as a comparison baseline."],
        ["DeiT (Touvron et al., 2021)", "Data-efficient through distillation from a teacher model, which adds a dependency that obscures the effect of self-supervision."],
        ["Swin Transformer (Liu et al., 2021)", "Hierarchical design adds architectural priors; a possible later extension, not the cleanest test of plain ViT behaviour."],
        ["ViT with DRLoc (Liu et al., 2021)", "Relevant data-efficient method; considered a candidate comparison rather than the primary approach."],
        ["Attention fusion ViT (Rahmat et al., 2024)", "Targets architecture design rather than label efficiency."]]))
    p("The papers use different datasets, tasks and settings, so their reported scores are not interchangeable, and no single model can be "
      "declared best. The chosen pair (pre-trained and conventional ViT) isolates one factor, the pre-training, which keeps the "
      "hypothesis testable and the interpretation clear.")
    h2("7.4 Connection to the dataset")
    p("HAM10000 provides roughly ten thousand dermatoscopic images, a small-to-moderate size for transformers, and an imbalance "
      "that leaves few examples of rare classes. All training images are available without labels for pre-training, while labelled "
      "subsets can be reduced to simulate scarce annotation. This matches the model's strength and the research problem. The "
      "justification is a reason to investigate the approach, not a claim that it will outperform CNNs or other transformers.")
    B.append(("refs", REFS))
    return B


def fig_counts(path):
    plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar([c[0] for c in CLASSES], [c[2] for c in CLASSES], color="black")
    ax.set_yscale("log"); ax.set_ylabel("Images")
    for i, c in enumerate(CLASSES):
        ax.text(i, c[2] * 1.1, f"{c[2]:,}", ha="center", fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def render_pptx(figdir, path):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    BLK, GRY, WHT = RGBColor(0, 0, 0), RGBColor(0x59, 0x59, 0x59), RGBColor(255, 255, 255)
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]; n = [0]

    def tb(sl, x, y, w, h, text, size=18, bold=False, color=BLK, font="Calibri"):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True
        for i, ln in enumerate(text if isinstance(text, list) else [text]):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = para.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold
            r.font.color.rgb = color; r.font.name = font
        return s

    def bullets(sl, items, x=0.7, y=1.7, w=11.9, size=20):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(4.9)); tf = s.text_frame; tf.word_wrap = True
        for i, t in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.space_after = Pt(12)
            r = para.add_run(); r.text = "▪  " + t; r.font.size = Pt(size); r.font.color.rgb = BLK; r.font.name = "Calibri"

    def slide(title):
        sl = prs.slides.add_slide(blank); n[0] += 1
        tb(sl, 0.7, 0.45, 12, 0.9, title, 30, True, font="Georgia")
        ln = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.3), Inches(11.9), Inches(0.03))
        ln.fill.solid(); ln.fill.fore_color.rgb = BLK; ln.line.fill.background()
        tb(sl, 0.7, 7.0, 10, 0.3, f"{md.SHORT}  |  {md.AUTHOR}  |  {md.SID}", 10, color=GRY)
        tb(sl, 11.6, 7.0, 1.0, 0.3, str(n[0]), 10, color=GRY)
        return sl

    def box(sl, x, y, w, h, text, dark=False, size=16):
        s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid(); s.fill.fore_color.rgb = BLK if dark else WHT; s.line.color.rgb = BLK; s.line.width = Pt(1.5)
        tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, ln in enumerate(text.split("\n")):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = PP_ALIGN.CENTER
            r = para.add_run(); r.text = ln; r.font.size = Pt(size); r.font.name = "Calibri"
            r.font.color.rgb = WHT if dark else BLK; r.font.bold = (i == 0)

    def table(sl, x, y, w, head, body, size=14):
        t = sl.shapes.add_table(len(body) + 1, len(head), Inches(x), Inches(y), Inches(w), Inches(0.45 * (len(body) + 1))).table
        for j, h in enumerate(head):
            c = t.cell(0, j); c.text = h; c.fill.solid(); c.fill.fore_color.rgb = BLK
            for para in c.text_frame.paragraphs:
                for r in para.runs: r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = WHT; r.font.name = "Calibri"
        for i, row in enumerate(body, 1):
            for j, v in enumerate(row):
                c = t.cell(i, j); c.text = str(v); c.fill.solid()
                c.fill.fore_color.rgb = WHT if i % 2 else RGBColor(0xF2, 0xF2, 0xF2)
                for para in c.text_frame.paragraphs:
                    for r in para.runs: r.font.size = Pt(size); r.font.color.rgb = BLK; r.font.name = "Calibri"

    s = prs.slides.add_slide(blank); n[0] += 1
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BLK
    tb(s, 0.9, 1.3, 11.5, 2.8, TITLE, 36, True, WHT, "Georgia")
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(4.3), Inches(3), Inches(0.04)); ln.fill.solid()
    ln.fill.fore_color.rgb = WHT; ln.line.fill.background()
    tb(s, 0.9, 4.5, 11.5, 2.2, ["Week 4: Research Framing and Proposal Development", f"{md.AUTHOR}  |  Student ID {md.SID}",
        "MAAI901 – Deep Learning for Visual Data  |  Instructor: Dr. Milan Dordevic", "University of Wollongong in Dubai"], 18, color=WHT)

    s = slide("Topic Description")
    bullets(s, ["Skin lesion classification from dermoscopy can support early detection of skin cancer.",
                "Labelled medical images are scarce and datasets are imbalanced: many common, few rare diagnoses.",
                "High overall accuracy can hide failures on rare, clinically important classes.",
                "Vision Transformers (ViTs) need large data; remedies include self-supervised pre-training.",
                "Focus: ViT reliability when labels and minority examples are both limited."])
    s = slide("Literature Review: Five Base Papers")
    table(s, 0.7, 1.6, 11.9, ["Paper", "Contribution", "Method (high level)", "Limitation"], [
        ["Liu et al. 2021", "ViTs on small data", "DRLoc auxiliary task", "General images, not dermoscopy"],
        ["Xiao et al. 2023", "Medical ViT pre-training", "MAE on 266,340 X-rays", "Chest X-ray only"],
        ["Ansari et al. 2024", "Imbalance + skin-tone bias", "MixUp, adaptive sampler", "Data efficiency not studied"],
        ["Rahmat et al. 2024", "Attention-fusion ViT", "3 attention types, ISIC 2017", "Reduced data not tested"],
        ["Shajimon et al. 2023", "Improved ViT", "Focal loss, 5-fold CV", "Binary task only"]], 13)
    SL = [
     ("Liu et al., 2021 (NeurIPS)", "Efficient Training of Visual Transformers with Small Datasets",
      ["Contribution: ViTs behave differently from CNNs on small data; improved results across several transformer architectures and datasets.",
       "Method: auxiliary self-supervised Dense Relative Localization (DRLoc) task added to supervised training.",
       "Limitation: general images, not medical; rare lesion classes not addressed."]),
     ("Xiao et al., 2023 (WACV)", "Delving into Masked Autoencoders for Multi-Label Thorax Disease Classification",
      ["Contribution: medical-image pre-training; ViT-S competitive with DenseNet-121 on three chest X-ray datasets.",
       "Method: MAE pre-training on 266,340 unlabelled chest X-rays (mask patches, reconstruct pixels).",
       "Limitation: chest X-rays, not dermoscopy; needs a large unlabelled collection."]),
     ("Ansari et al., 2024 (MICCAI)", "Algorithmic Fairness in Lesion Classification by Mitigating Class Imbalance and Skin Tone Bias",
      ["Contribution: addresses unequal lesion representation and skin-tone bias; stresses class-wise evaluation.",
       "Method: MixUp-based minority augmentation plus an adaptive sampler; ISIC 2018 and ASAN.",
       "Limitation: limited model configurations; data efficiency of transformers not examined."]),
     ("Rahmat et al., 2024 (MIUA)", "Enhancing Skin Lesion Classification: A Self-Attention Fusion Approach with Vision Transformer",
      ["Contribution: ViT on dermoscopy; abstract reports F1 improvement of more than 12% over the authors' baseline.",
       "Method: fuses scaled dot-product, multiplicative and additive attention; ISIC 2017.",
       "Limitation: one dataset; robustness with reduced data or severe imbalance not shown."]),
     ("Shajimon et al., 2023 (BIBM)", "An Improved Vision-Transformer Network for Skin Cancer Classification",
      ["Contribution: improved ViT, competitive with CNN-based methods and a soft-attention model.",
       "Method: modified ViT-B16, 224 x 224 input, augmentation, focal loss, five-fold cross-validation; ISIC 2017.",
       "Limitation: binary task; challenges with similar categories and darker skin tones."]),
    ]
    for k, (head, full, items) in enumerate(SL, 1):
        s = slide(f"Paper {k}: {head}")
        tb(s, 0.7, 1.45, 11.9, 0.8, full, 17, True, GRY)
        bullets(s, items, y=2.4, size=19)
    s = slide("Cross-paper Synthesis")
    bullets(s, ["Liu (2021) and Xiao (2023): self-supervision reduces label dependence, but not on skin images.",
                "Ansari, Rahmat, Shajimon: dermoscopy studies with different targets, data and tasks.",
                "Scores are not directly comparable across datasets and settings.",
                "Missing: behaviour as labelled data shrink, with class-wise performance."])
    s = slide("Research Gap")
    box(s, 0.7, 1.7, 11.9, 1.4, "Gap\nNo reviewed study evaluates self-supervised ViTs class-wise on multi-class dermoscopy as labelled data shrink.", True, 18)
    bullets(s, ["Liu: small data, general images.  Xiao: chest X-rays.", "Ansari: imbalance, not transformer data efficiency.",
                "Rahmat, Shajimon: full training size; Shajimon binary only.", "Need: macro-F1 and per-class recall, not accuracy alone."], y=3.4, size=18)
    s = slide("Problem Statement")
    box(s, 0.7, 1.8, 11.9, 3.4, PROBLEM, False, 20)
    s = slide("Dataset: HAM10000")
    table(s, 0.7, 1.6, 6.4, ["Class", "Images", "Share"], [[a, f"{c:,}", f"{100*c/10015:.1f}%"] for a, b, c in CLASSES], 13)
    pth = figdir / "fig_ham_counts.png"
    if pth.exists(): s.shapes.add_picture(str(pth), Inches(7.4), Inches(1.7), width=Inches(5.3))
    tb(s, 0.7, 5.6, 11.9, 1.2, "10,015 dermatoscopic images, 7 classes (Tschandl et al., 2018). Imbalanced, public, widely used. Limitation: curated, limited skin-tone diversity.", 15, color=GRY)
    s = slide("Research Hypothesis")
    box(s, 0.7, 1.7, 5.8, 2.6, "H1\nWith limited labels, a ViT with masked-image pre-training achieves higher macro-F1 than the same ViT trained from scratch.", True, 17)
    box(s, 6.8, 1.7, 5.8, 2.6, "H0\nThe pre-trained ViT does not achieve higher macro-F1 than the ViT trained from scratch.", False, 17)
    tb(s, 0.7, 4.8, 11.9, 1.6, "Macro-F1 weights every class equally; per-class recall shows whether rare diagnoses are missed.", 18, color=GRY)
    s = slide("Model Selection")
    box(s, 0.7, 1.7, 11.9, 1.2, "ViT-S/16 with masked image modelling pre-training (MAE / SimMIM style)", True, 20)
    bullets(s, ["Self-attention models global lesion cues: asymmetry, colour, border.", "Pre-training uses unlabelled images, so it does not depend on scarce labels.",
                "Compared with the same ViT trained from scratch to isolate the effect of pre-training."], y=3.3, size=18)
    s = slide("Alternatives Considered")
    table(s, 0.7, 1.6, 11.9, ["Alternative", "Why not primary"], [
        ["ResNet CNN", "Strong priors; kept as baseline"], ["DeiT", "Needs a teacher; obscures self-supervision effect"],
        ["Swin Transformer", "Extra architectural priors; later extension"], ["ViT + DRLoc", "Candidate comparison"],
        ["Attention-fusion ViT", "Architecture focus, not label efficiency"]], 14)
    s = slide("Scope and Expected Contribution")
    bullets(s, ["A focused research direction: data-efficient ViTs for imbalanced skin lesion data.",
                "Report excludes methodology, experiments and results, as required at this stage.",
                "Findings of a later study would be research evidence, not clinical recommendations."])
    s = slide("References")
    tb(s, 0.7, 1.55, 11.9, 5.3, ["Ansari et al. (2024). Algorithmic fairness in lesion classification. MICCAI.", "Dosovitskiy et al. (2021). An image is worth 16x16 words. ICLR.",
        "He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.", "Liu et al. (2021). Efficient training of visual transformers with small datasets. NeurIPS.",
        "Rahmat et al. (2024). Self-attention fusion approach with Vision Transformer. MIUA.", "Shajimon et al. (2023). An improved vision-transformer network. BIBM.",
        "Tschandl et al. (2018). The HAM10000 dataset. Scientific Data, 5, 180161.", "Xiao et al. (2023). Masked autoencoders for thorax disease classification. WACV.",
        "Xie et al. (2022). SimMIM. CVPR.   |   Touvron et al. (2021). DeiT. ICML.   |   Liu et al. (2021). Swin. ICCV.   |   He et al. (2016). ResNet. CVPR."], 14)
    prs.save(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="documents"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True); fd = out / "_fig"; fd.mkdir(exist_ok=True)
    fig_counts(fd / "fig_ham_counts.png")
    print("Topic description words:", wc(TOPIC))
    for t, ps in PAPERS:
        print(" ", t.split(":")[0], "words:", sum(wc(x) for x in ps))
    B = build(fd)
    base = f"Week4_ResearchFraming_{md.SID}"
    md.render_docx(B, out / f"{base}.docx"); md.render_pdf(B, out / f"{base}.pdf"); render_pptx(fd, out / f"{base}.pptx")
    print("Created:", *sorted(p.name for p in out.glob("Week4*")), sep="\n  ")


if __name__ == "__main__":
    main()
