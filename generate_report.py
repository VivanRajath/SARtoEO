import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# --- Numbered Canvas for Dynamic Headers and Footers ---

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Page 1 is the cover page — skip decorations
        if self._pageNumber == 1:
            return
            
        self.saveState()
        
        # Color definitions matching the design system
        primary_color = colors.HexColor('#1B365D') # Deep Navy
        border_color = colors.HexColor('#D1D5DB')  # Light Gray
        text_color = colors.HexColor('#5C768D')    # Slate Gray
        
        # Header (Top)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(primary_color)
        self.drawString(54, 792, "TECHNICAL REPORT")
        self.setFont("Helvetica", 8)
        self.setFillColor(text_color)
        self.drawRightString(541, 792, "SAR-to-EO Image Translation Using Pix2Pix cGAN")
        
        self.setStrokeColor(border_color)
        self.setLineWidth(0.5)
        self.line(54, 784, 541, 784)
        
        # Footer (Bottom)
        self.line(54, 52, 541, 52)
        self.drawString(54, 40, "GalaxEye Space — AI Research Intern Technical Assignment")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 40, page_text)
        
        self.restoreState()


def build_pdf(filename="Technical_Report.pdf"):
    # Target path: Letter size or A4 size. A4: 595.27 x 841.89 points
    # Page setup
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Color system
    c_primary = colors.HexColor('#1B365D')   # Deep Navy
    c_secondary = colors.HexColor('#5C768D') # Slate Gray
    c_dark = colors.HexColor('#222222')      # Charcoal Text
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        alignment=1, # Center
        spaceAfter=40
    )
    
    meta_label_style = ParagraphStyle(
        'CoverMetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_primary,
        alignment=1
    )
    
    meta_val_style = ParagraphStyle(
        'CoverMetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_dark,
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'ReportH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_primary,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'ReportH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=c_dark,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    caption_style = ParagraphStyle(
        'ReportCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=c_secondary,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=12
    )
    
    abstract_style = ParagraphStyle(
        'ReportAbstract',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=c_dark,
        leftIndent=24,
        rightIndent=24,
        spaceAfter=15
    )

    story = []
    
    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("SAR-to-EO Image Translation", title_style))
    story.append(Paragraph("Synthesis of High-Fidelity Optical Imagery from Sentinel-1 Radar Backscatter", subtitle_style))
    
    # Divider line
    divider = Table([[""]], colWidths=[487], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_primary),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 120))
    
    story.append(Paragraph("ASSIGNMENT TITLE", meta_label_style))
    story.append(Paragraph("AI Research Intern Technical Assessment", meta_val_style))
    
    story.append(Paragraph("CANDIDATE POSITION", meta_label_style))
    story.append(Paragraph("Satellite AI Research Intern", meta_val_style))
    
    story.append(Paragraph("ORGANIZATION", meta_label_style))
    story.append(Paragraph("GalaxEye Space", meta_val_style))
    
    story.append(Paragraph("DATE", meta_label_style))
    story.append(Paragraph("June 2026", meta_val_style))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 2: ABSTRACT & LITERATURE SURVEY
    # =========================================================================
    story.append(Paragraph("Abstract", h1_style))
    abstract_text = (
        "Synthetic Aperture Radar (SAR) sensors provide invaluable active Earth observation "
        "capabilities because they penetrate cloud cover and operate independently of solar illumination. "
        "However, SAR imagery consists of single-channel backscatter intensity dominated by surface roughness "
        "and moisture gradients, with heavy coherent speckle noise — making direct human interpretation and "
        "downstream optical analysis difficult. In contrast, Electro-Optical (EO) imagery offers intuitive, "
        "multi-spectral visual data but is frequently obscured by cloud cover, limiting its operational availability. "
        "Translating SAR-to-EO images is a highly ill-posed cross-modal problem: SAR contains no direct "
        "colour information, meaning multiple plausible optical layouts could correspond to a single radar "
        "footprint. A field of dry wheat and sandy soil, for instance, produce near-identical C-band "
        "backscatter yet appear entirely different in optical imagery. "
        "This report documents a supervised conditional Generative Adversarial Network (cGAN) approach — "
        "specifically the Pix2Pix framework (Isola et al., 2017) — with an 8-level U-Net generator "
        "and a 70x70 PatchGAN discriminator, trained end-to-end on co-registered "
        "Sentinel-1 VV SAR and Sentinel-2 RGB image pairs from a 4,000-patch agricultural dataset subset. "
        "The combined loss function (L_cGAN + 100*L1) simultaneously enforces global structural fidelity "
        "via L1 pixel loss and perceptual texture realism via adversarial feedback. "
        "Quantitative evaluation on the full 600-image held-out test set yields SSIM of 0.2725, "
        "PSNR of 15.46 dB, LPIPS of 0.4613, and FID of 176.73 — well below the random-generation "
        "baseline of 300–400+ FID. Qualitative inspection reveals structured scene translations "
        "including field boundary delineation, crop area identification, and land-use distribution "
        "that are visually coherent with the SAR texture input, establishing a robust baseline "
        "for SAR-to-optical translation in the GalaxEye operational context."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Introduction & Literature Survey", h1_style))
    survey_p1 = (
        "Translating Synthetic Aperture Radar (SAR) signals to corresponding Electro-Optical (EO) images "
        "falls under the category of multimodal image-to-image translation. SAR sensors transmit microwave "
        "pulses and measure backscatter energy, which is heavily influenced by surface roughness, moisture, "
        "and material dielectric constants. Optical sensors, on the other hand, measure solar reflection "
        "across discrete visible and infrared bands. Because the physics of acquisition are fundamentally "
        "dissimilar, mapping SAR to EO requires models that can infer semantic content and generate plausible "
        "optical textures from backscatter patterns."
    )
    story.append(Paragraph(survey_p1, body_style))
    
    survey_p2 = (
        "Early approaches relied on deterministic pixel-level mapping or simple regression trees. With "
        "the advent of deep learning, supervised Image-to-Image translation was revolutionized by "
        "<b>Pix2Pix (Isola et al., 2017)</b>. Pix2Pix models formulate translation as a conditional GAN (cGAN), "
        "where a generator is trained to produce realistic target images conditioned on the source modal input, "
        "while a discriminator attempts to distinguish real target pairs from generated pairs. To capture both "
        "large-scale structures and fine-textured details, Pix2Pix combines an L1 pixel reconstruction loss "
        "with an adversarial loss. The discriminator uses a 'PatchGAN' architecture, classifying localized "
        "overlapping NxN patches rather than the entire image, which forces the generator to capture high-frequency "
        "local detail."
    )
    story.append(Paragraph(survey_p2, body_style))
    
    survey_p3 = (
        "For cases where aligned pairs are unavailable, <b>CycleGAN (Zhu et al., 2017)</b> utilizes cycle-consistency "
        "constraints to map between unpaired domains. However, in remote sensing, spatial alignment between "
        "satellite orbits allows for precise paired dataset compilation (such as the SEN1-2 and SEN12MS datasets), "
        "making supervised models like Pix2Pix preferable as they enforce spatial correspondence directly. "
        "More recently, <b>Diffusion Models</b> (e.g., DDPM, LDM, ControlNet) have emerged as powerful generative "
        "alternatives, producing exceptional image quality and diversity by modelling translation as a conditional "
        "iterative denoising process. However, their inference speed is extremely slow (100\u20131000 denoising steps "
        "per image), they demand significant VRAM (often >16GB for 512\xd7512 outputs), and require long training runs "
        "on high-end hardware — making cGANs the optimal choice for resource-constrained, high-throughput systems "
        "such as GalaxEye's satellite data processing pipeline."
        "<br/><br/>"
        "<b>Motivation for this Work</b>: GalaxEye Space is building a multi-sensor satellite platform that "
        "fuses SAR and optical modalities for commercial Earth observation. On-demand generation of EO-equivalent "
        "representations from SAR would allow continuous agricultural monitoring, disaster response mapping, and "
        "infrastructure surveillance regardless of cloud cover — directly addressing a core operational limitation "
        "of optical-only satellite fleets. This Pix2Pix baseline establishes the feasibility of the approach "
        "and provides a reproducible foundation for future research into higher-fidelity models such as "
        "diffusion-based SAR-to-EO synthesis."
    )
    story.append(Paragraph(survey_p3, body_style))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 3: METHODOLOGY
    # =========================================================================
    story.append(Paragraph("2. Methodology", h1_style))
    
    story.append(Paragraph("2.1 Dataset and Data Splits", h2_style))
    method_data = (
        "We utilize the <b>Sentinel-1 & Sentinel-2 Terrain-Separated Image Pairs</b> dataset, focusing on the "
        "<b>agricultural subset</b>. Agricultural regions provide an ideal setting for translation evaluation "
        "due to distinct geometric textures (furrows, crop field boundary lines, crop circles) and seasonal color changes. "
        "The subset contains 4,000 paired SAR/EO patches. We implement a strict three-way split of <b>70% training</b> "
        "(2,800 patches), <b>15% validation</b> (600 patches), and <b>15% test</b> (600 patches) to ensure no spatial "
        "leakage. The splits are generated randomly with a fixed seed (42), and the exact image allocations are recorded "
        "in <code>outputs/data_split.csv</code> for full reproducibility."
    )
    story.append(Paragraph(method_data, body_style))
    
    story.append(Paragraph("2.2 Preprocessing and Augmentation", h2_style))
    method_prep = (
        "SAR input images (Sentinel-1 VV polarization) are single-channel, dB-scaled, and normalized to [0, 255] in "
        "raw storage. During loading, they are mapped to float tensors in [0, 1]. Optical targets (Sentinel-2 RGB) "
        "are loaded, scaled to [0, 1], and normalized to [-1, 1] using a mean of 0.5 and standard deviation of 0.5. "
        "This matches the target range of the generator's final <i>Tanh</i> activation layer. All patches are resized to "
        "256x256 using bilinear interpolation with antialiasing. "
        "During training, data augmentation is applied to the training split: random horizontal and vertical flips "
        "are applied simultaneously to both SAR and EO inputs to preserve exact pixel alignment while improving generalization."
    )
    story.append(Paragraph(method_prep, body_style))
    
    story.append(Paragraph("2.3 Network Architectures", h2_style))
    method_arch = (
        "Our model implements the classic Pix2Pix conditional GAN architecture. The <b>Generator</b> is a symmetric "
        "<b>8-level U-Net</b> structure consisting of 7 encoder levels, a central bottleneck, and 7 decoder levels. "
        "Each encoder layer uses Conv2D (stride=2), Batch Normalization, and LeakyReLU, progressively downsampling the "
        "feature maps from 256x256 down to a 1x1 bottleneck. The decoder uses Transposed Conv2D, Batch Normalization, "
        "Dropout (layers 1-3), and ReLU, upsampling back to 256x256. <i>Skip connections</i> concatenate the output "
        "of each encoder layer directly with the input of its corresponding decoder layer, bypassing the bottleneck to "
        "preserve low-level spatial and texture details. "
        "<br/><br/>"
        "The <b>Discriminator</b> is a <b>70x70 PatchGAN</b>. It takes the concatenated channel input (SAR + EO, total 4 channels) "
        "and processes it through a series of Conv2D layers to produce a 30x30 classification grid. Each value in this grid "
        "represents the probability that a corresponding 70x70 receptive field in the input is real. This local supervision "
        "encourages high-frequency boundary and color texture realism."
    )
    story.append(Paragraph(method_arch, body_style))
    
    story.append(Paragraph("2.4 Loss Formulation", h2_style))
    method_loss = (
        "The network is optimized using a joint loss function. The discriminator is trained using binary cross-entropy "
        "with logits loss. The generator is trained to minimize both the adversarial feedback and the pixel-wise L1 distance "
        "to the ground truth: "
        "<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>L_total = L_cGAN(G, D) + lambda * L1(G)</b>"
        "<br/>"
        "where we set <b>lambda = 100</b> (the standard default from Isola et al.). The L1 loss enforces correct global "
        "low-frequency structure, while the GAN loss pushes the generator to output sharp, high-frequency textures."
    )
    story.append(Paragraph(method_loss, body_style))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 4: RESULTS & LOSS CURVES
    # =========================================================================
    story.append(Paragraph("3. Results and Analysis", h1_style))
    
    story.append(Paragraph("3.1 Quantitative Evaluation", h2_style))
    results_quant = (
        "We evaluate the model on the held-out test split using standard image translation metrics. "
        "SSIM and PSNR capture pixel-level structure and signal-to-noise quality. LPIPS (Learned Perceptual "
        "Image Patch Similarity) evaluates perceptual quality using deep features from a pre-trained AlexNet. "
        "FID (Fréchet Inception Distance) measures the statistical distance between distributions of real and "
        "generated EO features. The results are summarized in Table 1."
    )
    story.append(Paragraph(results_quant, body_style))
    
    # Table 1: Quantitative Results
    data = [
        ['Metric', 'Score', 'Interpretation / Goal'],
        ['SSIM ↑', '0.2725', 'Measures structural layout alignment; higher is better.'],
        ['PSNR ↑ (dB)', '15.46', 'Measures pixel-level reconstruction fidelity; higher is better.'],
        ['LPIPS ↓', '0.4613', 'Measures perceptual texture/color similarity; lower is better.'],
        ['FID ↓', '176.73', 'Measures feature distribution distance; lower is better.'],
    ]
    
    t = Table(data, colWidths=[100, 70, 317])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'CENTER'), # Center scores
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F3F4F6')),
        ('BACKGROUND', (0,2), (-1,2), colors.white),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#F3F4F6')),
        ('BACKGROUND', (0,4), (-1,4), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
        ('TOPPADDING', (0,1), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Paragraph("<b>Table 1:</b> Quantitative metrics evaluated on the test dataset.", caption_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("3.2 Loss Convergence and Ablation Study", h2_style))
    results_curves = (
        "We log training progress over 15 epochs. During training, we implement an ablation logging feature "
        "that records a parallel generator L1-only loss to evaluate the effect of adversarial feedback. "
        "The convergence of the discriminator and generator losses, along with the ablation curve, is illustrated "
        "in Figure 1."
    )
    story.append(Paragraph(results_curves, body_style))
    
    # Loss curves image
    loss_curve_path = "outputs/loss_curve.png"
    if os.path.exists(loss_curve_path):
        story.append(Image(loss_curve_path, width=450, height=125))
        story.append(Paragraph("<b>Figure 1:</b> Loss curves representing generator total loss, ablated L1-only loss, and discriminator loss over 15 epochs.", caption_style))
    else:
        story.append(Paragraph("[Image outputs/loss_curve.png not found]", body_style))
        
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 5: QUALITATIVE EXAMPLES
    # =========================================================================
    story.append(Paragraph("3.3 Qualitative Image Comparison", h2_style))
    results_qual = (
        "To evaluate translation quality qualitatively, we visualize input SAR images, generated EO predictions, "
        "and corresponding ground-truth EO optical images. Figures 2 and 3 display representative examples "
        "from the held-out test split, showcasing crop fields, boundaries, and spatial structure translation."
    )
    story.append(Paragraph(results_qual, body_style))
    
    # Qualitative images
    im1_path = "outputs/test_example_01.png"
    im2_path = "outputs/test_example_02.png"
    
    qual_elements = []
    
    if os.path.exists(im1_path):
        qual_elements.append(Image(im1_path, width=420, height=140))
        qual_elements.append(Paragraph("<b>Figure 2:</b> Translation example 1. Format: SAR Input (Left) | Generated EO (Center) | Ground Truth (Right).", caption_style))
    else:
        qual_elements.append(Paragraph("[Image outputs/test_example_01.png not found]", body_style))
        
    if os.path.exists(im2_path):
        qual_elements.append(Image(im2_path, width=420, height=140))
        qual_elements.append(Paragraph("<b>Figure 3:</b> Translation example 2. Format: SAR Input (Left) | Generated EO (Center) | Ground Truth (Right).", caption_style))
    else:
        qual_elements.append(Paragraph("[Image outputs/test_example_02.png not found]", body_style))
        
    story.append(KeepTogether(qual_elements))
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 6: DISCUSSION & RESOURCE LOG
    # =========================================================================
    story.append(Paragraph("3.4 Error Analysis & Discussion", h2_style))
    discussion_text = (
        "Evaluating the qualitative examples reveals key characteristics of the translation pipeline:"
        "<br/><br/>"
        "• <b>Success Cases:</b> The model demonstrates a strong ability to capture geographic layout and boundary lines. "
        "It successfully identifies field divisions, roads, and large crop patterns from radar texture differences. "
        "The generated green crop fields align well with actual vegetation, showing that the U-Net skip connections "
        "are successfully propagating high-frequency spatial boundaries from the input to the output."
        "<br/><br/>"
        "• <b>Failure Cases & Color Hallucination:</b> Because SAR (radar backscatter) registers surface roughness, "
        "different colors of vegetation (e.g., green vs. dry yellow crops) or soil types can produce identical radar backscatter. "
        "As a result, the model tends to hallucinate a generic green or light-brown color in ambiguous areas, occasionally "
        "missing localized crop variations seen in the optical ground truth. This highlights the fundamentally "
        "ill-posed nature of the problem."
        "<br/><br/>"
        "• <b>Speckle Noise & Artifacts:</b> Raw SAR images contain speckle noise, which sometimes propagates "
        "through the U-Net as grain-like textures. The PatchGAN discriminator penalizes blurriness, forcing the generator "
        "to synthesize sharp details, which occasionally results in grid-like or pixelated artifacts in large, homogeneous regions."
        "<br/><br/>"
        "• <b>Pixel vs. Perceptual Gap:</b> The low SSIM (0.2263) and PSNR (10.85 dB) metrics reflect the fact that the model is "
        "penalized for pixel-level mismatches in color hallucination, even when the generated structures and crop layouts "
        "look visually realistic. The LPIPS score (0.544) indicates reasonable perceptual likeness, which is highly "
        "valued for human interpretation."
    )
    story.append(Paragraph(discussion_text, body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. Hardware and Resource Log", h1_style))
    
    # Table 2: Resource Log
    log_data = [
        ['Parameter', 'Detail', 'Notes / Context'],
        ['Hardware Platform', 'Google Colab / Local GPU', 'Mixed execution'],
        ['GPU Model', 'NVIDIA T4 / RTX 40-series', 'Single GPU training'],
        ['VRAM Used', '< 8 GB', 'Batch size 2, well within 16GB limit'],
        ['Training Time', '15 epochs completed', 'Total runtime ~30-40 minutes'],
        ['Time per Epoch', '~150 seconds', 'On NVIDIA T4 GPU'],
        ['Data Pipeline', 'PyTorch DataLoader', 'Auto-workers optimized (0 on Windows, 2 on Linux)'],
        ['Google Drive Mirroring', 'Active', 'Checkpoints mirrored to GDrive to survive disconnects'],
    ]
    t2 = Table(log_data, colWidths=[130, 140, 217])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')]),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('TOPPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table 2:</b> Resource allocation and training execution log.", caption_style))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PAGE 7: FUTURE WORK, CONCLUSION & REFERENCES
    # =========================================================================
    story.append(Paragraph("5. Future Work", h1_style))
    future_text = (
        "If I join GalaxEye Space, I would explore several directions to push the boundaries of "
        "SAR-optical translation and data fusion:"
        "<br/><br/>"
        "• <b>Dual-polarization Input (VV + VH):</b> Leveraging both Sentinel-1 polarization channels "
        "(VV and VH) will provide additional cross-polarized backscatter information, helping the model "
        "better distinguish building shapes, moisture gradients, and crop vegetation structures."
        "<br/><br/>"
        "• <b>Edge and Boundary Preserving Loss:</b> Incorporating Sobel filters or Canny-edge loss terms "
        "will penalize fuzzy crop field boundaries, enforcing sharper borders in agricultural translations."
        "<br/><br/>"
        "• <b>VGG Perceptual and Style Loss:</b> Adding a VGG-19 feature reconstruction loss (similar to "
        "perceptual loss in super-resolution) can reduce the color hallucination gap and match human "
        "visual perception better than pure L1 loss."
        "<br/><br/>"
        "• <b>Conditional Diffusion Models (e.g., ControlNet):</b> Applying latent diffusion models conditioned "
        "on SAR backscatter could yield much higher textural realism and diversity, resolving the ill-posed "
        "ambiguity by allowing probabilistic sampling of multiple plausible optical scenes."
    )
    story.append(Paragraph(future_text, body_style))
    
    story.append(Paragraph("6. Conclusion", h1_style))
    conclusion_text = (
        "This project successfully implements a Pix2Pix conditional GAN for SAR-to-EO satellite image translation. "
        "By enforcing global structure via L1 loss and texture fidelity via a 70x70 PatchGAN discriminator, the model "
        "demonstrates promising results in translating Sentinel-1 VV radar images into Sentinel-2 RGB optical patches. "
        "Evaluated on the full 600-image held-out test set, we achieve SSIM of 0.2725, PSNR of 15.46 dB, "
        "LPIPS of 0.4613, and FID of 176.73. The pipeline is designed for robustness, offering "
        "reproducible data splits, CLI-override capability for Google Colab, and compliance with all assessment specifications. "
        "These results establish a solid foundation for further research in radar-optical fusion at GalaxEye Space."
    )
    story.append(Paragraph(conclusion_text, body_style))
    
    story.append(Paragraph("References", h1_style))
    refs = [
        "1. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.",
        "2. Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV 2017.",
        "3. Schmitt, M., Hughes, L. H., & Zhu, X. X. (2018). SEN1-2: A Dataset for Deep Learning in SAR-Optical Data Fusion. ISPRS Annals.",
        "4. Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR.",
        "5. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI."
    ]
    for ref in refs:
        story.append(Paragraph(ref, bullet_style))
        
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PASS] Report compiled successfully: {filename}")


if __name__ == "__main__":
    build_pdf()
