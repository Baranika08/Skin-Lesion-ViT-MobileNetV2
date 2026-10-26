"""Generate the final report (DOCX + PDF) and the presentation (PPTX) from results/results.json.
Black-and-white professional style. Run:  python make_documents.py"""
import json, datetime, argparse
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

AUTHOR, SID = "Baranika V G", "1671212"
TITLE = "Data-Efficient Vision Transformers for Skin Lesion Classification under Limited and Imbalanced Training Data"
SHORT = "Data-Efficient ViTs for Skin Lesion Classification"
SUBTITLE = "Project Report"
EXTRA = []
NAMES = {"cnn": "MobileNetV2 baseline", "mobilenet_v2": "MobileNetV2 baseline",
         "vit_scratch": "ViT (scratch)", "vit_ssl": "ViT + masked pretraining",
         "vit_ssl_weighted": "ViT + masked pretraining + weighted loss"}
REFS = [
    "Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (ICLR 2021).",
    "He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).",
    "Xie, Z., Zhang, Z., Cao, Y., et al. (2022). SimMIM: A simple framework for masked image modeling. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022).",
    "Liu, Y., Sangineto, E., Bi, W., Sebe, N., Lepri, B., & De Nadai, M. (2021). Efficient training of visual transformers with small datasets. Advances in Neural Information Processing Systems (NeurIPS 2021).",
    "Xiao, J., Bai, Y., Yuille, A., & Zhou, Z. (2023). Delving into masked autoencoders for multi-label thorax disease classification. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023).",
    "Ansari, F., Chakraborti, T., & Das, S. (2024). Algorithmic fairness in lesion classification by mitigating class imbalance and skin tone bias. Medical Image Computing and Computer Assisted Intervention (MICCAI 2024).",
    "Rahmat, I. H., Gan, J. Q., & Raza, H. (2024). Enhancing skin lesion classification: A self-attention fusion approach with Vision Transformer. Medical Image Understanding and Analysis (MIUA 2024).",
    "Shajimon, G. M., Ufumaka, I., & Raza, H. (2023). An improved vision-transformer network for skin cancer classification. IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2023).",
    "Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations (ICLR 2019).",
    "Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS 2019).",
]


# ---------------------------------------------------------------- analysis helpers
def fmt(m, s=None):
    return f"{m:.3f}" if s is None else f"{m:.3f} ± {s:.3f}"


def table_f1(res):
    rows, meta = res["rows"], res["meta"]
    head = ["Method"] + [f"{f*100:g}% labels" for f in meta["fractions"]]
    body = []
    for m in meta["methods"]:
        line = [NAMES[m]]
        for f in meta["fractions"]:
            v = [r["macro_f1"] for r in rows if r["method"] == m and abs(r["fraction"] - f) < 1e-9]
            line.append(fmt(np.mean(v), np.std(v)) if v else "n/a")
        body.append(line)
    return head, body


def verdict(res):
    rows, meta = res["rows"], res["meta"]
    out = []
    for f in meta["fractions"]:
        sel = lambda m: sorted([r for r in rows if r["method"] == m and abs(r["fraction"] - f) < 1e-9],
                               key=lambda r: r["seed"])
        a, b = sel("vit_ssl"), sel("vit_scratch")
        d = [x["macro_f1"] - y["macro_f1"] for x, y in zip(a, b)]
        out.append(dict(f=f, ssl=np.mean([x["macro_f1"] for x in a]), scr=np.mean([y["macro_f1"] for y in b]),
                        diff=float(np.mean(d)), wins=int(sum(v > 0 for v in d)), n=len(d)))
    pos = sum(v["diff"] > 0 for v in out)
    maj = all(v["wins"] > v["n"] / 2 for v in out if v["diff"] > 0) and pos > 0
    if pos == len(out) and maj:
        txt = "consistent with H1"
    elif pos == 0:
        txt = "not supported (H0 is retained)"
    else:
        txt = "mixed: the advantage depends on the label budget"
    return out, txt


def best_at(res, f):
    rows = res["rows"]
    sc = {m: np.mean([r["macro_f1"] for r in rows if r["method"] == m and abs(r["fraction"] - f) < 1e-9])
          for m in res["meta"]["methods"]}
    b = max(sc, key=sc.get)
    return NAMES[b], sc[b]


# ---------------------------------------------------------------- content
def build(res, figdir):
    figdir = Path(figdir)
    B = []
    h1 = lambda t: B.append(("h1", t)); h2 = lambda t: B.append(("h2", t))
    p = lambda t: B.append(("p", t)); ul = lambda l: B.append(("ul", l))
    note = lambda t: B.append(("note", t))
    fig = lambda f, c, w=5.5: B.append(("fig", str(figdir / f), c, w)) if (figdir / f).exists() else None
    have = res is not None and res["rows"] and res.get("complete", True)
    meta = res["meta"] if have else {}
    syn = have and meta["dataset"] == "synthetic"

    h1("Abstract")
    p("Deep learning models for skin lesion classification are usually trained on datasets that are both small and "
      "strongly imbalanced, so a high overall accuracy can hide poor performance on rare diagnoses. This project "
      "investigates whether a Vision Transformer (ViT) that is first pre-trained without labels by masked-patch "
      "reconstruction keeps its macro-F1 better than a conventionally trained ViT and a convolutional baseline "
      "when only a fraction of the labels is available. A complete, reproducible pipeline was implemented in "
      "PyTorch: lesion-grouped data splits, label-budget experiments, self-supervised pre-training, fine-tuning, "
      "class-wise evaluation, and automatic report generation.")
    if have:
        v, txt = verdict(res)
        f0 = min(meta["fractions"])
        bn, bs = best_at(res, f0)
        p(f"Experiments used {meta['dataset_label']} with {len(meta['seeds'])} random seed(s) and label budgets of "
          f"{', '.join(f'{f*100:g}%' for f in meta['fractions'])}. At the smallest budget ({f0*100:g}%), the best "
          f"method by mean test macro-F1 was {bn} ({bs:.3f}). The comparison of the data-efficient ViT with the "
          f"conventional ViT was {txt}.")
        if syn:
            note("IMPORTANT: this report was produced from the built-in synthetic proxy dataset to verify the pipeline "
                 "end-to-end. The numbers do not describe real skin lesions and must not be interpreted clinically. "
                 "Run  python run_all.py --dataset ham10000  to regenerate every table, figure and document from HAM10000.")
    else:
        note("Experimental results are not included yet. Run  python run_all.py  to execute the experiments and "
             "regenerate this report with real measured numbers.")
    h1("1. Introduction")
    p("Skin lesion classification from dermoscopic images can support clinical decision-making, but labelled medical "
      "images are costly to obtain and datasets contain many examples of common diagnoses and few of rare ones. A "
      "model can therefore reach a high overall accuracy while missing minority classes, which is the situation "
      "where errors matter most. Vision Transformers (Dosovitskiy et al., 2021) model relations between image patches "
      "with self-attention, but they have weaker built-in image priors than convolutional networks and are known to "
      "need considerable data (Liu et al., 2021).")
    p("This project studies the combined effect of limited labels and class imbalance. Rather than claiming that one "
      "architecture is universally better, it asks a narrow, testable question and evaluates it with class-sensitive "
      "metrics. The system is a research prototype and not a diagnostic tool.")
    h1("2. Related Work")
    p("Liu et al. (2021) showed that ViTs behave differently from CNNs on small datasets and proposed Dense Relative "
      "Localization (DRLoc), an auxiliary self-supervised task that encourages spatial reasoning. The study uses general "
      "computer-vision data, so its benefit for dermoscopy and imbalanced classes is not established.")
    p("Xiao et al. (2023) pre-trained ViTs with masked autoencoders on 266,340 unlabelled chest X-rays and obtained "
      "results competitive with DenseNet-121 on thoracic disease classification. The finding supports medical-image "
      "self-supervised pre-training but concerns another modality and relies on a large unlabelled collection.")
    p("Ansari et al. (2024) addressed class imbalance and skin-tone bias in lesion classification with MixUp-based "
      "augmentation for minority classes and an adaptive sampler, evaluated on ISIC 2018 and ASAN. Imbalance and skin-tone "
      "bias are related but distinct problems, and the evaluation covered a limited set of model configurations.")
    p("Rahmat et al. (2024) combined scaled dot-product, multiplicative and additive attention in a ViT for skin lesion "
      "classification on ISIC 2017 and report an F1 improvement of more than 12% over their baseline. Robustness when "
      "training data are strongly reduced was not examined.")
    p("Shajimon et al. (2023) modified a ViT-B16 with extra layers and regularisation, used augmentation, focal loss and "
      "five-fold cross-validation on ISIC 2017, and obtained competitive results against CNN-based methods for a "
      "binary task. Broader multi-class settings and darker skin tones remain open.")
    p("Synthesis. The five studies address data efficiency, medical pre-training, imbalance and attention design "
      "separately. None of them evaluates, for multi-class dermoscopy, how a self-supervised ViT behaves as the number "
      "of labels shrinks while minority-class performance is tracked. That intersection is the gap addressed here; "
      "the claim is deliberately limited, as other studies may touch parts of it.")
    h1("3. Problem Statement and Hypothesis")
    p("Problem statement: It is unclear how reliably Vision Transformer classifiers recognise minority diagnostic classes "
      "of skin lesions when both the labelled training data and the minority-class examples are limited. This project "
      "evaluates whether masked-patch self-supervised pre-training improves class-wise reliability under such conditions.")
    p("H1: With a limited labelled-data budget, a ViT initialised by masked-patch pre-training achieves a higher test "
      "macro-F1 than the same ViT trained from scratch on a multi-class, imbalanced skin lesion dataset.")
    p("H0: Under the same budget, the pre-trained ViT does not achieve a higher macro-F1 than the ViT trained from scratch.")
    h1("4. Dataset")
    if have and not syn:
        p("HAM10000 (Tschandl et al., 2018) contains 10,015 dermatoscopic images in seven diagnostic categories: actinic "
          "keratoses/intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis (bkl), dermatofibroma "
          "(df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (vasc). Melanocytic nevi dominate the dataset "
          "(about two thirds of the images), which makes it a realistic imbalanced benchmark. Several images can show the "
          "same lesion, so splits are grouped by lesion identifier to prevent leakage between training and test sets.")
    elif have:
        p("For pipeline verification only, a synthetic dermoscopy-like dataset is generated. Each of the seven classes "
          "(named after HAM10000 categories) has an overlapping colour, size, boundary-irregularity and texture "
          "distribution, and class frequencies follow HAM10000 proportions. It exists so the entire pipeline can be run "
          "in minutes without any download; it carries no medical meaning. The real target dataset is HAM10000 "
          "(Tschandl et al., 2018), whose use requires accepting its licence terms.")
    else:
        p("The target dataset is HAM10000 (Tschandl et al., 2018): 10,015 dermatoscopic images in seven imbalanced "
          "diagnostic categories. A synthetic proxy can be used for a quick pipeline check.")
    if have:
        p(f"After splitting (stratified by class, grouped by lesion), the training, validation and test sets contain "
          f"{meta['n_train']}, {meta['n_val']} and {meta['n_test']} images respectively (total {meta['n_total']}).")
        fig("fig_class_distribution.png", "Figure 1. Class distribution of the training and test splits (log scale).", 4.8)
        fig("fig_samples.png", "Figure 2. Example images per class (rows).", 3.9)
    h1("5. Methodology")
    h2("5.1 Pre-processing and splits")
    p("Images are resized to a square resolution and scaled to [-1, 1]. Data are divided into train, validation and test "
      "sets (about 71%, 14%, 14%) with stratified group splitting so every lesion appears in one split only. Training "
      "augmentation consists of random flips, transposition (all eight dihedral orientations) and mild brightness jitter; "
      "validation and test images are never augmented.")
    h2("5.2 Label-budget protocol")
    fr = ", ".join(f"{f*100:g}%" for f in meta["fractions"]) if have else "10%, 25%, 50%, 100%"
    p(f"To simulate scarce annotation, the labelled training set is subsampled to {fr} of the training images, stratified "
      "by class (at least two images per class). The validation set used for checkpoint selection is reduced in the same "
      "proportion so that no method benefits from a large labelled validation pool. The test set is always complete and "
      "is never used for tuning. The number of epochs is scaled for small budgets (up to four times) so that the number "
      "of optimiser steps remains comparable.")
    h2("5.3 Models")
    vp = f"{meta['params']['vit']:,}" if have else "about 0.8 million"
    cp = f"{meta['params']['cnn']:,}" if have else "about 0.3 million"
    cfg = meta.get("vit_cfg", dict(img=64, patch=8, dim=128, depth=4, heads=4))
    ul([f"ViT (scratch): patch size {cfg['patch']}, embedding width {cfg['dim']}, {cfg['depth']} pre-norm transformer blocks, "
        f"{cfg['heads']} heads, learnable class token and positional embeddings ({vp} parameters), randomly initialised.",
        f"ViT + masked pre-training: the same architecture, first pre-trained on the unlabelled training images by "
        f"hiding 60% of the patches and reconstructing their normalised pixels (a SimMIM-style objective; Xie et al., "
        f"2022; He et al., 2022), then fine-tuned with labels. Labels are never used in pre-training and validation/test "
        f"images are never seen.",
        "ViT + masked pre-training + weighted loss: as above, with a class-weighted cross-entropy (inverse class frequency) "
        "to counter imbalance.",
        f"CNN baseline: four convolution-batch-norm-ReLU-max-pool blocks with global average pooling ({cp} parameters)."])
    h2("5.4 Training and evaluation")
    p("All models are optimised with AdamW (Loshchilov & Hutter, 2019) using a one-cycle learning-rate schedule, weight "
      "decay 0.05, batch size 64 and gradient clipping, and are implemented in PyTorch (Paszke et al., 2019). The checkpoint "
      "with the best validation macro-F1 is evaluated once on the test set. The primary metric is macro-F1, which weights "
      "all classes equally; per-class recall, accuracy and balanced accuracy are reported as supporting measures. Every "
      "experiment is repeated over several seeds, which also change the data split, and mean ± standard deviation is reported.")
    h1("6. Experimental Setup")
    if have:
        ul([f"Dataset: {meta['dataset_label']}.",
            f"Image size {meta['img']}x{meta['img']}, patch {meta['patch']}; fine-tuning epochs {meta['epochs']} "
            f"(scaled for small budgets); pre-training epochs {meta['pretrain_epochs']}.",
            f"Seeds: {len(meta['seeds'])}; label budgets: {fr}.",
            f"Device: {meta['device']}; PyTorch {meta['torch']}; total run time {meta.get('runtime_min','?')} minutes."])
    else:
        p("The exact configuration is inserted automatically after the experiments are run.")
    h1("7. Results")
    if have:
        if syn:
            note("Synthetic proxy data: results verify the pipeline only and are not evidence about real skin lesions.")
        head, body = table_f1(res)
        B.append(("table", "Table 1. Test macro-F1 (mean ± standard deviation over seeds).", head, body))
        fig("fig_macro_f1.png", "Figure 3. Test macro-F1 as the labelled-data budget increases.", 5.0)
        v, txt = verdict(res)
        rows_ = [[f"{x['f']*100:g}%", fmt(x["ssl"]), fmt(x["scr"]), f"{x['diff']:+.3f}", f"{x['wins']}/{x['n']}"] for x in v]
        B.append(("table", "Table 2. Data-efficient ViT versus conventional ViT (test macro-F1).",
                  ["Labels", "ViT + pretraining", "ViT (scratch)", "Difference", "Seeds won"], rows_))
        p(f"Hypothesis check. Across the label budgets, the outcome is {txt}. The differences and the number of seeds in "
          f"which the pre-trained model was better are listed in Table 2. With only {len(meta['seeds'])} seed(s), no "
          f"formal significance claim is made; the evidence should be read as indicative.")
        fig("fig_recall.png", f"Figure 4. Per-class test recall at the smallest budget ({min(meta['fractions'])*100:g}% labels).", 5.6)
        fig("fig_confusion.png", "Figure 5. Row-normalised confusion matrices at the smallest budget (summed over seeds).", 5.6)
        fig("fig_pretrain_loss.png", "Figure 6. Masked-patch reconstruction loss during self-supervised pre-training (seed 0).", 3.6)
        f0, f1_ = min(meta["fractions"]), max(meta["fractions"])
        bn0, bs0 = best_at(res, f0); bn1, bs1 = best_at(res, f1_)
        p(f"Observations. With {f0*100:g}% of the labels the best method was {bn0} (macro-F1 {bs0:.3f}); with all labels "
          f"it was {bn1} ({bs1:.3f}). The per-class recall and confusion matrices show which diagnoses lose the most "
          f"performance when labels are scarce, which overall accuracy would not reveal.")
    else:
        note("No results yet. Execute python run_all.py to fill this chapter automatically.")
    h1("8. Discussion and Limitations")
    ul(["The study uses small models and low-resolution images to keep experiments reproducible on modest hardware; "
        "conclusions may change with larger models, higher resolution or large-scale pre-training.",
        "Self-supervised pre-training here uses only the unlabelled images of the training split; external unlabelled "
        "dermoscopy data could behave differently.",
        "A single dataset is used; generalisation across devices, hospitals and skin tones is not demonstrated, and "
        "skin-tone annotations are not available in HAM10000 (Ansari et al., 2024).",
        "A small number of seeds limits statistical power; differences within one standard deviation should not be over-interpreted.",
        "The prototype is not a medical device. Any clinical use would need external validation, regulatory approval and "
        "human oversight."])
    h1("9. Conclusion")
    if have:
        v, txt = verdict(res)
        p(f"This project built and evaluated a reproducible pipeline for studying data-efficient Vision Transformers under "
          f"limited and imbalanced labels. In the experiments reported here, the hypothesis was {txt}. "
          + ("Because the run used synthetic data, the result demonstrates that the pipeline works but says nothing about real lesions. "
             if syn else "")
          + "Future work includes more seeds, higher resolution, larger pre-training corpora, alternative imbalance "
            "strategies and external test sets with skin-tone annotations.")
    else:
        p("Conclusions are generated from the measured results after the experiments have been run.")
    h1("10. Reproducibility")
    p("The complete code, configuration and instructions are in the project folder. python run_all.py --dataset synthetic "
      "runs the pipeline end-to-end without downloads; python run_all.py --dataset ham10000 --data-root path/to/HAM10000 "
      "reproduces the real-data study. All figures, tables, the report, the PDF and the slides are produced from "
      "results/results.json, so documents always match the measured numbers.")
    B.append(("refs", REFS))
    return B


# ---------------------------------------------------------------- DOCX renderer
def render_docx(B, path):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = Document()
    for s in d.sections:
        s.page_width, s.page_height = Inches(8.27), Inches(11.69)
        s.top_margin = s.bottom_margin = s.left_margin = s.right_margin = Inches(1)
    def setfont(style, size, bold=False, italic=False):
        style.font.name = "Times New Roman"; style.font.size = Pt(size); style.font.bold = bold
        style.font.italic = italic; style.font.color.rgb = RGBColor(0, 0, 0)
        rf = style.element.rPr.rFonts
        for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(att)) is not None:
                del rf.attrib[qn(att)]
        for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(att), "Times New Roman")
    n = d.styles["Normal"]; setfont(n, 12); n.paragraph_format.line_spacing = 1.5
    n.paragraph_format.space_after = Pt(6)
    for name, sz in (("Heading 1", 14), ("Heading 2", 12)):
        st = d.styles[name]; setfont(st, sz, True); st.paragraph_format.space_before = Pt(14)
        st.paragraph_format.space_after = Pt(6); st.paragraph_format.keep_with_next = True
    # footer page number
    fp = d.sections[0].footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run()
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)
    # title page
    def centered(text, size, bold=False, before=0, italic=False):
        para = d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(before); run = para.add_run(text)
        run.font.size = Pt(size); run.bold = bold; run.italic = italic; return para
    centered("UNIVERSITY OF WOLLONGONG IN DUBAI", 13, True, 60)
    centered("Master’s in Applied Artificial Intelligence", 12)
    centered("MAAI901 – Deep Learning for Visual Data", 12)
    for e in EXTRA:
        centered(e, 12)
    centered(TITLE, 20, True, 90)
    centered(SUBTITLE, 14, False, 30, True)
    centered(AUTHOR, 13, True, 90)
    centered(f"Student ID: {SID}", 12)
    centered(datetime.date.today().strftime("%d %B %Y"), 12)
    d.add_page_break()
    heads = [b for b in B if b[0] in ("h1",)]
    d.add_heading("Table of Contents", 1)
    for b in heads:
        d.add_paragraph(b[1]).paragraph_format.space_after = Pt(0)
    d.add_paragraph("References")
    d.add_page_break()
    def shade(cell, color):
        tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)
    for b in B:
        k = b[0]
        if k == "h1":
            d.add_heading(b[1], 1)
        elif k == "h2":
            d.add_heading(b[1], 2)
        elif k == "p":
            d.add_paragraph(b[1]).alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif k == "ul":
            for t in b[1]:
                d.add_paragraph(t, style="List Bullet")
        elif k == "note":
            t = d.add_table(rows=1, cols=1); t.style = "Table Grid"; c = t.cell(0, 0); shade(c, "EEEEEE")
            c.text = ""; run = c.paragraphs[0].add_run(b[1]); run.bold = True; run.font.size = Pt(11)
            c.paragraphs[0].paragraph_format.line_spacing = 1.15; d.add_paragraph()
        elif k == "fig":
            para = d.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.add_run().add_picture(b[1], width=Inches(b[3])); para.paragraph_format.keep_with_next = True
            cp = d.add_paragraph(); cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(b[2]); r.italic = True; r.font.size = Pt(10); cp.paragraph_format.line_spacing = 1.0
        elif k == "table":
            cp = d.add_paragraph(); r = cp.add_run(b[1]); r.italic = True; r.font.size = Pt(10)
            cp.paragraph_format.keep_with_next = True; cp.paragraph_format.line_spacing = 1.0
            t = d.add_table(rows=1, cols=len(b[2])); t.style = "Table Grid"; t.alignment = 1
            for i, h in enumerate(b[2]):
                c = t.rows[0].cells[i]; c.text = ""; run = c.paragraphs[0].add_run(h); run.bold = True
                run.font.size = Pt(10); shade(c, "D9D9D9")
            for row in b[3]:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""; run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(10)
            for ri, row in enumerate(t.rows):
                for c in row.cells:
                    c.paragraphs[0].paragraph_format.line_spacing = 1.0
                    c.paragraphs[0].paragraph_format.keep_with_next = ri < len(t.rows) - 1
            d.add_paragraph()
        elif k == "refs":
            d.add_page_break(); d.add_heading("References", 1)
            for t in b[1]:
                para = d.add_paragraph(t); para.paragraph_format.left_indent = Inches(.5)
                para.paragraph_format.first_line_indent = Inches(-.5); para.paragraph_format.line_spacing = 1.15
    d.save(path)


# ---------------------------------------------------------------- PDF renderer
def render_pdf(B, path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
                                    Image, Table, TableStyle, ListFlowable, ListItem, KeepTogether)
    from reportlab.platypus.tableofcontents import TableOfContents
    from PIL import Image as PI
    F = "Times-Roman"; FB = "Times-Bold"; FI = "Times-Italic"
    S = dict(
        body=ParagraphStyle("body", fontName=F, fontSize=11.5, leading=17.5, alignment=TA_JUSTIFY, spaceAfter=6),
        H1=ParagraphStyle("H1", fontName=FB, fontSize=14, leading=18, spaceBefore=14, spaceAfter=8, keepWithNext=1),
        H2=ParagraphStyle("H2", fontName=FB, fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, keepWithNext=1),
        cap=ParagraphStyle("cap", fontName=FI, fontSize=9.5, leading=12, alignment=TA_CENTER, spaceAfter=10),
        cell=ParagraphStyle("cell", fontName=F, fontSize=9.5, leading=12),
        cellb=ParagraphStyle("cellb", fontName=FB, fontSize=9.5, leading=12),
        note=ParagraphStyle("note", fontName=FB, fontSize=10, leading=13),
        ref=ParagraphStyle("ref", fontName=F, fontSize=10.5, leading=14, leftIndent=28, firstLineIndent=-28, spaceAfter=5),
        c=lambda sz, b=False, it=False, sb=0: ParagraphStyle("c", fontName=FB if b else (FI if it else F),
                                                            fontSize=sz, leading=sz * 1.3, alignment=TA_CENTER, spaceBefore=sb))

    class Doc(BaseDocTemplate):
        def __init__(self, fn):
            super().__init__(fn, pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch,
                             title=TITLE, author=AUTHOR)
            self.addPageTemplates([PageTemplate("f", [Frame(inch, inch, A4[0] - 2 * inch, A4[1] - 2 * inch)],
                                                onPage=self.foot)])
        def foot(self, c, d):
            if d.page > 1:
                c.setFont(F, 10); c.drawCentredString(A4[0] / 2, 0.55 * inch, str(d.page - 1))
        def afterFlowable(self, fl):
            if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2") and getattr(fl, "toc", True):
                self.notify("TOCEntry", (0 if fl.style.name == "H1" else 1, fl.getPlainText(), self.page - 1))

    st = []
    st += [Spacer(1, 70), Paragraph("UNIVERSITY OF WOLLONGONG IN DUBAI", S["c"](13, True)),
           Paragraph("Master’s in Applied Artificial Intelligence", S["c"](12)),
           Paragraph("MAAI901 – Deep Learning for Visual Data", S["c"](12))] + \
          [Paragraph(escape(e), S["c"](12)) for e in EXTRA] + [
           Spacer(1, 80), Paragraph(escape(TITLE), S["c"](20, True)), Spacer(1, 24),
           Paragraph(SUBTITLE, S["c"](14, it=True)), Spacer(1, 90),
           Paragraph(AUTHOR, S["c"](13, True)), Paragraph(f"Student ID: {SID}", S["c"](12)),
           Paragraph(datetime.date.today().strftime("%d %B %Y"), S["c"](12)), PageBreak()]
    cp = Paragraph("Table of Contents", S["H1"]); cp.toc = False; st.append(cp)
    toc = TableOfContents(); toc.levelStyles = [
        ParagraphStyle("t0", fontName=F, fontSize=11.5, leading=18, leftIndent=0),
        ParagraphStyle("t1", fontName=F, fontSize=10.5, leading=15, leftIndent=18)]
    st += [toc, PageBreak()]
    for b in B:
        k = b[0]
        if k in ("h1", "h2"):
            st.append(Paragraph(escape(b[1]), S["H1" if k == "h1" else "H2"]))
        elif k == "p":
            st.append(Paragraph(escape(b[1]), S["body"]))
        elif k == "ul":
            st.append(ListFlowable([ListItem(Paragraph(escape(t), S["body"]), leftIndent=18) for t in b[1]],
                                   bulletType="bullet", start="•", leftIndent=18))
        elif k == "note":
            t = Table([[Paragraph(escape(b[1]), S["note"])]], colWidths=[A4[0] - 2 * inch])
            t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, colors.black),
                                   ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEEEE")),
                                   ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            st += [t, Spacer(1, 8)]
        elif k == "fig":
            w, h = PI.open(b[1]).size; wd = b[3] * inch
            st.append(KeepTogether([Image(b[1], width=wd, height=wd * h / w), Paragraph(escape(b[2]), S["cap"])]))
        elif k == "table":
            data = [[Paragraph(escape(str(h)), S["cellb"]) for h in b[2]]] + \
                   [[Paragraph(escape(str(v)), S["cell"]) for v in r] for r in b[3]]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            st.append(KeepTogether([Paragraph(escape(b[1]), S["cap"]), t, Spacer(1, 10)]))
        elif k == "refs":
            st += [PageBreak(), Paragraph("References", S["H1"])] + [Paragraph(escape(t), S["ref"]) for t in b[1]]
    Doc(str(path)).multiBuild(st)


# ---------------------------------------------------------------- PPTX renderer
def render_pptx(res, figdir, path):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    figdir = Path(figdir)
    BLK, GRY, WHT = RGBColor(0, 0, 0), RGBColor(0x59, 0x59, 0x59), RGBColor(255, 255, 255)
    prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    have = res is not None and res["rows"]
    meta = res["meta"] if have else {}
    syn = have and meta["dataset"] == "synthetic"
    count = [0]

    def tb(sl, x, y, w, h, text, size=18, bold=False, color=BLK, align=PP_ALIGN.LEFT, font="Calibri", anchor=None):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True
        if anchor: tf.vertical_anchor = anchor
        lines = text if isinstance(text, list) else [text]
        for i, ln in enumerate(lines):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.alignment = align; r = para.add_run(); r.text = ln
            r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = font
        return s

    def bullets(sl, x, y, w, h, items, size=20):
        s = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = s.text_frame; tf.word_wrap = True
        for i, t in enumerate(items):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.space_after = Pt(12); r = para.add_run(); r.text = "▪  " + t
            r.font.size = Pt(size); r.font.color.rgb = BLK; r.font.name = "Calibri"

    def slide(title):
        sl = prs.slides.add_slide(blank); count[0] += 1
        tb(sl, 0.7, 0.45, 12, 0.9, title, 32, True, font="Georgia")
        ln = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), Inches(1.3), Inches(11.9), Inches(0.03))
        ln.fill.solid(); ln.fill.fore_color.rgb = BLK; ln.line.fill.background()
        tb(sl, 0.7, 7.0, 10, 0.3, f"{SHORT}  |  {AUTHOR}  |  {SID}", 10, color=GRY)
        tb(sl, 11.6, 7.0, 1.0, 0.3, str(count[0]), 10, color=GRY, align=PP_ALIGN.RIGHT)
        return sl

    def pic(sl, name, x, y, w=None, h=None):
        pth = figdir / name
        if pth.exists():
            kw = {}
            if w: kw["width"] = Inches(w)
            if h: kw["height"] = Inches(h)
            sl.shapes.add_picture(str(pth), Inches(x), Inches(y), **kw)

    def box(sl, x, y, w, h, text, dark=False, size=15):
        s = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        s.fill.solid(); s.fill.fore_color.rgb = BLK if dark else WHT; s.line.color.rgb = BLK; s.line.width = Pt(1.5)
        tf = s.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, ln in enumerate(text.split("\n")):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); para.alignment = PP_ALIGN.CENTER
            r = para.add_run(); r.text = ln; r.font.size = Pt(size); r.font.name = "Calibri"
            r.font.color.rgb = WHT if dark else BLK; r.font.bold = (i == 0)

    def pending(sl):
        box(sl, 1.5, 2.2, 10.3, 2.4, "Results pending\nExperiments have not been run yet.\nRun  python run_all.py  (then rebuild) to fill this slide with measured results.", False, 20)

    def arrow(sl, x, y):
        a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(0.5), Inches(0.4))
        a.fill.solid(); a.fill.fore_color.rgb = BLK; a.line.fill.background()

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

    # 1 title (inverted)
    sl = prs.slides.add_slide(blank); count[0] += 1
    bg = sl.background.fill; bg.solid(); bg.fore_color.rgb = BLK
    tb(sl, 0.9, 1.5, 11.5, 2.6, TITLE, 38, True, WHT, font="Georgia")
    ln = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(4.3), Inches(3), Inches(0.04)); ln.fill.solid()
    ln.fill.fore_color.rgb = WHT; ln.line.fill.background()
    tb(sl, 0.9, 4.5, 11, 1.8, [AUTHOR + "  |  Student ID " + SID, "MAAI901 – Deep Learning for Visual Data",
                               "University of Wollongong in Dubai"], 20, color=WHT)
    if syn:
        tb(sl, 0.9, 6.5, 11.5, 0.5, "Pipeline-verification run on synthetic data – not real skin-lesion results", 14, True, WHT)

    s = slide("Problem and Motivation")
    bullets(s, 0.7, 1.7, 11.9, 4.8, [
        "Skin lesion classification from dermoscopy can support clinical decisions.",
        "Labelled medical images are scarce; rare diagnoses have very few examples.",
        "High overall accuracy can hide failures on minority classes.",
        "Vision Transformers (ViTs) are data-hungry compared with CNNs.",
        "Goal: test whether self-supervised pre-training makes ViTs more reliable with few labels."])
    s = slide("Research Question and Hypothesis")
    box(s, 0.7, 1.7, 11.9, 1.3, "Problem statement\nHow reliably do ViT classifiers recognise minority skin-lesion classes when labels are limited?", size=16)
    box(s, 0.7, 3.3, 5.8, 1.9, "H1\nWith limited labels, a ViT with masked-patch pre-training achieves higher macro-F1 than the same ViT trained from scratch.", dark=True, size=15)
    box(s, 6.8, 3.3, 5.8, 1.9, "H0\nThe pre-trained ViT does not achieve higher macro-F1 than the ViT trained from scratch.", size=15)
    tb(s, 0.7, 5.5, 11.9, 1.0, "Primary metric: macro-F1 (all classes weighted equally). Supporting: per-class recall, accuracy, balanced accuracy.", 17, color=GRY)
    s = slide("Related Work and Gap")
    table(s, 0.7, 1.6, 11.9, ["Study", "Contribution", "Open question"], [
        ["Liu et al. 2021", "DRLoc auxiliary task for ViTs on small data", "Not dermoscopy"],
        ["Xiao et al. 2023", "MAE pre-training on chest X-rays", "Other modality, large unlabelled set"],
        ["Ansari et al. 2024", "MixUp + adaptive sampler for imbalance / skin tone", "Data efficiency not studied"],
        ["Rahmat et al. 2024", "Attention-fusion ViT, ISIC 2017", "Strongly reduced data not tested"],
        ["Shajimon et al. 2023", "Improved ViT, focal loss, ISIC 2017", "Binary task only"]], 14)
    tb(s, 0.7, 5.0, 11.9, 1.4, "Gap: no study examines a self-supervised ViT for multi-class dermoscopy as the label budget shrinks, with class-wise performance tracked.", 18, True)
    s = slide("Dataset")
    if have and not syn:
        bullets(s, 0.7, 1.7, 6.3, 4.8, ["HAM10000: 10,015 dermoscopic images, 7 classes.", "Strong imbalance (melanocytic nevi ≈ 67%).", "Lesion-grouped split prevents leakage.", f"Train/val/test: {meta['n_train']}/{meta['n_val']}/{meta['n_test']}"], 18)
    elif have:
        bullets(s, 0.7, 1.7, 6.3, 4.8, ["Synthetic proxy with HAM10000 class proportions.", "Used only to verify the pipeline quickly.", "Real target: HAM10000 (Tschandl et al., 2018).", f"Train/val/test: {meta['n_train']}/{meta['n_val']}/{meta['n_test']}"], 18)
    else:
        bullets(s, 0.7, 1.7, 6.3, 4.8, ["Target: HAM10000, 10,015 images, 7 imbalanced classes.", "Lesion-grouped, class-stratified splits."], 18)
    pic(s, "fig_class_distribution.png", 7.2, 1.9, w=5.4)
    s = slide("Approach")
    for i, (t, d) in enumerate([("Unlabelled\nimages", False), ("Masked-patch\npre-training", True), ("Fine-tune with\nlabel budget", False), ("Test:\nmacro-F1", True)]):
        box(s, 0.7 + i * 3.1, 2.4, 2.4, 1.5, t, d, 17)
        if i < 3: arrow(s, 3.2 + i * 3.1, 2.95)
    tb(s, 0.7, 4.6, 11.9, 1.8, ["Label budgets: 10%, 25%, 50%, 100% of training labels (class-stratified).",
                                "Pre-training uses training images only; validation and test never seen.",
                                "Checkpoint chosen on validation macro-F1; test used once."], 17, color=GRY)
    s = slide("Models Compared")
    table(s, 0.7, 1.7, 11.9, ["Model", "Description"], [
        ["CNN baseline", "4 conv blocks + global pooling"],
        ["ViT (scratch)", "Patch ViT, 4 blocks, random initialisation"],
        ["ViT + masked pretraining", "Same ViT, SimMIM-style pre-training, then fine-tuned"],
        ["ViT + pretraining + weighted loss", "Adds inverse-frequency class weights"]], 16)
    s = slide("Experimental Protocol")
    bullets(s, 0.7, 1.7, 11.9, 4.8, [
        "Splits: ~71/14/14, stratified by class, grouped by lesion.",
        "Augmentation: flips, transposition, brightness jitter (training only).",
        "Optimiser: AdamW, one-cycle schedule, weight decay 0.05.",
        "Metrics: macro-F1 (primary), per-class recall, accuracy, balanced accuracy.",
        (f"Seeds: {len(meta['seeds'])}; mean ± std reported." if have else "Several seeds; mean ± std reported.")])
    s = slide("Results: Macro-F1 vs Label Budget")
    if have:
        pic(s, "fig_macro_f1.png", 0.6, 1.7, w=7.0)
        v, txt = verdict(res)
        bullets(s, 7.9, 1.8, 4.9, 4.3, [f"At {x['f']*100:g}% labels: ViT+pretraining {x['ssl']:.3f} vs scratch {x['scr']:.3f} ({x['diff']:+.3f})" for x in v] , 14)
        if syn: tb(s, 7.9, 6.3, 4.9, 0.6, "Synthetic data: pipeline check only.", 13, True, GRY)
    else:
        pending(s)
    s = slide("Results: Per-class Recall")
    if have: pic(s, "fig_recall.png", 1.2, 1.6, w=10.8)
    else: pending(s)
    s = slide("Results: Confusion Matrices")
    if have: pic(s, "fig_confusion.png", 1.6, 1.55, h=5.3)
    else: pending(s)
    s = slide("Hypothesis Check")
    if have:
        v, txt = verdict(res)
        table(s, 0.7, 1.7, 11.9, ["Labels", "ViT + pretraining", "ViT (scratch)", "Difference", "Seeds won"],
              [[f"{x['f']*100:g}%", fmt(x["ssl"]), fmt(x["scr"]), f"{x['diff']:+.3f}", f"{x['wins']}/{x['n']}"] for x in v], 16)
        tb(s, 0.7, 5.2, 11.9, 1.4, f"Outcome: {txt}. Few seeds → indicative, not a formal significance test.", 19, True)
    else:
        pending(s)
    s = slide("Limitations and Responsible Use")
    bullets(s, 0.7, 1.7, 11.9, 4.8, [
        "Small models, low resolution, one dataset; generalisation not shown.",
        "HAM10000 has no skin-tone labels; fairness cannot be assessed here.",
        "Few seeds limit statistical power.",
        "Research prototype only – not a diagnostic or medical device."])
    s = slide("Conclusion and Future Work")
    bullets(s, 0.7, 1.7, 11.9, 4.8, [
        "Built a reproducible pipeline: data → pre-training → fine-tuning → class-wise evaluation → documents.",
        (f"Hypothesis outcome: {verdict(res)[1]}." if have else "Hypothesis outcome follows from measured results."),
        "Next: more seeds, higher resolution, larger pre-training corpus, external test sets, imbalance strategies."])
    s = slide("How to Run")
    tb(s, 0.7, 1.7, 11.9, 4.5, ["pip install -r requirements.txt",
                                "python run_all.py --preset quick            # 10-minute check",
                                "python run_all.py                           # full synthetic run",
                                "python run_all.py --dataset ham10000 --data-root data/HAM10000",
                                "python demo.py --image path/to/lesion.jpg"], 18, font="Consolas")
    s = slide("References")
    tb(s, 0.7, 1.6, 11.9, 5.2, ["Dosovitskiy et al. (2021). An image is worth 16x16 words. ICLR.",
        "He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.",
        "Xie et al. (2022). SimMIM: a simple framework for masked image modeling. CVPR.",
        "Liu et al. (2021). Efficient training of visual transformers with small datasets. NeurIPS.",
        "Xiao et al. (2023). Delving into masked autoencoders for thorax disease classification. WACV.",
        "Ansari et al. (2024). Algorithmic fairness in lesion classification. MICCAI.",
        "Rahmat et al. (2024). Self-attention fusion approach with Vision Transformer. MIUA.",
        "Shajimon et al. (2023). An improved vision-transformer network for skin cancer classification. BIBM.",
        "Tschandl et al. (2018). The HAM10000 dataset. Scientific Data, 5, 180161."], 16)
    prs.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--out", default="documents")
    a = ap.parse_args()
    rp = Path(a.results)
    res = json.loads(rp.read_text()) if rp.exists() else None
    figdir = rp.parent / "figures"
    if res and res["rows"]:
        from src.figures import make_result_figures
        make_result_figures(res, figdir)
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    B = build(res, figdir)
    render_docx(B, out / f"Project_Report_{SID}.docx")
    render_pdf(B, out / f"Project_Report_{SID}.pdf")
    render_pptx(res, figdir, out / f"Project_Presentation_{SID}.pptx")
    print("Created:", *sorted(p.name for p in out.glob("*")), sep="\n  ")


if __name__ == "__main__":
    main()
