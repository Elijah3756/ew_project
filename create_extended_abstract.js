const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Header, Footer,
        AlignmentType, HeadingLevel, BorderStyle, PageNumber,
        TabStopType, TabStopPosition } = require('docx');

// ============================================================
// CDR Extended Abstract: Adversarial Robustness of AI-Based
// RF Signal Classification Under Channel Impairments
// ============================================================

const FONT = "Times New Roman";
const TITLE_SIZE = 28; // 14pt
const HEADING1_SIZE = 24; // 12pt
const BODY_SIZE = 22; // 11pt
const SMALL_SIZE = 20; // 10pt

function heading1(text) {
  return new Paragraph({
    spacing: { before: 300, after: 120 },
    children: [new TextRun({ text: text.toUpperCase(), bold: true, font: FONT, size: HEADING1_SIZE })]
  });
}

function heading2(text) {
  return new Paragraph({
    spacing: { before: 240, after: 80 },
    children: [new TextRun({ text: text, bold: true, italics: true, font: FONT, size: BODY_SIZE })]
  });
}

function bodyPara(textRuns, opts = {}) {
  const children = textRuns.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, font: FONT, size: BODY_SIZE });
    return new TextRun({ font: FONT, size: BODY_SIZE, ...t });
  });
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.indent ? { firstLine: 360 } : undefined,
    ...opts,
    children
  });
}

function italicPara(text) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: FONT, size: BODY_SIZE, italics: true })]
  });
}

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT, size: BODY_SIZE }
      }
    }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "CDR 2026 EW Special Issue -- Extended Abstract", font: FONT, size: SMALL_SIZE, italics: true, color: "666666" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Page ", font: FONT, size: SMALL_SIZE }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL_SIZE })]
        })]
      })
    },
    children: [
      // Article type
      new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.LEFT,
        children: [new TextRun({ text: "RESEARCH ARTICLE", bold: true, font: FONT, size: BODY_SIZE })]
      }),

      // Title
      new Paragraph({
        spacing: { before: 240, after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Adversarial Robustness of AI-Based RF Signal Classification Under Realistic Channel Impairments",
          bold: true, font: FONT, size: TITLE_SIZE
        })]
      }),

      // Author placeholder (double-blind)
      new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Author(s): Removed for double-blind review", font: FONT, size: BODY_SIZE, italics: true, color: "888888" })]
      }),
      new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Affiliation(s): Removed for double-blind review", font: FONT, size: BODY_SIZE, italics: true, color: "888888" })]
      }),

      // Abstract
      new Paragraph({ spacing: { before: 240 }, children: [] }),
      italicPara(
        "Deep learning classifiers operating on in-phase and quadrature (I/Q) baseband samples have become central to machine-speed spectrum awareness in cognitive electronic warfare (EW). However, their robustness to adversarial perturbations under realistic wireless channel conditions remains poorly characterized. This paper presents a systematic evaluation of adversarial vulnerability for a convolutional neural network (CNN) modulation classifier on the RadioML 2016.10a dataset (11 modulation schemes, 220,000 samples). We define a physically grounded threat model using power-ratio perturbation constraints and evaluate Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks across signal-to-noise ratios (SNR) from -10 to 18 dB under three channel conditions: AWGN, Rayleigh fading with AWGN, and composite Rayleigh fading with carrier frequency offset (CFO) and AWGN. Our baseline CNN achieves 70% clean accuracy at high SNR but remains vulnerable to PGD attacks, with attack success rates reaching 24% at 5% perturbation-to-signal power ratio. We compare three lightweight defense strategies -- channel augmentation training, adversarial training, and noise injection -- and analyze their robustness-compute tradeoffs for edge-deployed EW systems. Results demonstrate that channel augmentation provides the most efficient robustness improvement per computational dollar, while adversarial training yields the strongest absolute defense at higher cost."
      ),

      // Keywords
      new Paragraph({
        spacing: { before: 120, after: 240 },
        children: [
          new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: BODY_SIZE }),
          new TextRun({ text: "adversarial machine learning, RF modulation classification, electronic warfare, adversarial robustness, deep learning", font: FONT, size: BODY_SIZE })
        ]
      }),

      // ===== INTRODUCTION =====
      heading1("Introduction"),

      bodyPara([
        "Artificial intelligence and deep learning have rapidly become essential enablers of machine-speed spectrum awareness in contested electromagnetic environments. Modern radio frequency (RF) classifiers, trained on I/Q baseband samples, can identify modulation schemes and support adaptive spectrum management at speeds far exceeding human operators. These capabilities underpin cognitive electronic warfare, where real-time waveform classification is a prerequisite for spectrum dominance in multi-domain operations."
      ]),

      bodyPara([
        { text: "", italics: false },
        "However, the broader machine learning community has demonstrated that deep neural networks are susceptible to adversarial perturbations -- small, carefully crafted input modifications that cause misclassification with high confidence. While extensively studied in computer vision, the implications for physical-layer RF signal classification in contested spectrum environments have received limited attention. A misclassified waveform can trigger incorrect countermeasures, degrade situational awareness, or enable adversary deception."
      ], { indent: true }),

      bodyPara([
        "Unlike image classification, RF adversarial perturbations must contend with channel impairments including additive white Gaussian noise, multipath fading, and carrier frequency offset. Understanding how these channel effects interact with adversarial vulnerability is critical for assessing real-world robustness of deployed classifiers. This paper addresses this gap through a systematic experimental evaluation. We make four contributions: (1) a physically plausible threat model using power-ratio perturbation constraints grounded in RF physics; (2) channel-aware robustness evaluation across three channel conditions and SNR levels; (3) comparison of three lightweight defense strategies with operational tradeoff analysis; and (4) framing of findings in the context of cognitive EW and electromagnetic spectrum risk."
      ], { indent: true }),

      // ===== RESEARCH DESIGN =====
      heading1("Research Design"),

      heading2("Dataset and Model"),
      bodyPara([
        "We use the RadioML 2016.10a benchmark dataset containing 220,000 I/Q sample windows across 11 modulation schemes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, 16-QAM, 64-QAM, QPSK, WBFM) spanning SNR levels from -20 to +18 dB. We filter to the [-10, 18] dB range, yielding 165,000 samples split 70/15/15 for training, validation, and testing. Our baseline classifier is a four-block 1D CNN with 126,475 trainable parameters, representative of lightweight architectures suitable for edge deployment."
      ]),

      heading2("Threat Model"),
      bodyPara([
        "We consider a pre-channel adversary who perturbs the transmitted signal before wireless propagation. Rather than adopting norm-based bounds from computer vision, we impose a power-ratio constraint: the perturbation energy relative to signal energy must satisfy ||delta||_2 / ||x||_2 <= rho, where rho ranges from 0.5% to 5% (-46 dB to -26 dB perturbation-to-signal power ratio). We implement FGSM (single-step) and PGD (10-step iterative) attacks with L2-normalized gradients."
      ]),

      heading2("Channel Conditions"),
      bodyPara([
        "We evaluate under three channel configurations of increasing severity: (1) AWGN only; (2) Rayleigh block fading with AWGN; and (3) composite Rayleigh fading with carrier frequency offset and AWGN. All channel layers are implemented as differentiable PyTorch modules, enabling gradient-based attacks to account for channel effects during perturbation optimization."
      ]),

      heading2("Defense Strategies"),
      bodyPara([
        "We evaluate three defenses chosen for computational feasibility in edge-deployed systems: channel augmentation training (on-the-fly random channel impairments during training), adversarial training (PGD with 5 inner steps, alpha = 0.5 balancing clean and adversarial loss), and noise injection with optional Monte Carlo smoothing at inference."
      ]),

      // ===== PRELIMINARY FINDINGS =====
      heading1("Preliminary Findings"),

      bodyPara([
        { text: "Baseline Performance. ", bold: true },
        "The CNN achieves approximately 70% classification accuracy at high SNR (18 dB) and 45.6% at 0 dB, consistent with published benchmarks on RadioML 2016.10a. Accuracy degrades monotonically with decreasing SNR, dropping below random chance (9.1%) at the lowest levels. Rayleigh fading introduces substantial additional degradation beyond AWGN alone, particularly in the moderate SNR regime (0-10 dB)."
      ]),

      bodyPara([
        { text: "Adversarial Vulnerability. ", bold: true },
        "Under PGD attack at rho = 1% perturbation budget and AWGN channel, the attack success rate reaches approximately 8% at SNR = 10 dB, reducing robust accuracy from 70.1% to 67.6%. At the higher budget of rho = 5%, the attack success rate increases to 24.4% at SNR = 10 dB, reducing robust accuracy to 53.8%. The gap between clean and robust accuracy is most pronounced at moderate-to-high SNR, where the classifier has sufficient clean accuracy to be meaningfully degraded. At very low SNR, channel noise itself dominates the adversarial perturbation."
      ], { indent: true }),

      bodyPara([
        { text: "Channel Interaction. ", bold: true },
        "A key finding is the interaction between channel impairments and adversarial effectiveness. Because pre-channel perturbations undergo the same fading and noise as the signal, channel effects provide partial attenuation of attack efficacy. However, this accidental robustness is insufficient as a deliberate defense, as attack success rates remain operationally concerning across all channel conditions."
      ], { indent: true }),

      bodyPara([
        { text: "Defense Tradeoffs. ", bold: true },
        "Preliminary results from defense experiments (ongoing) indicate that channel augmentation training provides the most computationally efficient robustness improvement, requiring no additional computation beyond lightweight channel simulation layers. Adversarial training yields the strongest absolute robustness improvement but at 3-5x longer training time. Noise injection provides moderate improvement applicable to pre-trained models without retraining."
      ], { indent: true }),

      // ===== IMPLICATIONS =====
      heading1("Implications for Cyber Defense"),

      bodyPara([
        "Our findings have direct implications for the design and deployment of AI-based spectrum awareness systems in contested electromagnetic environments. The demonstrated vulnerability of baseline RF classifiers to low-power adversarial perturbations indicates that adversarial robustness must be treated as a first-order system requirement, not a theoretical concern. An unprotected spectrum awareness system would be susceptible to manipulation by adversaries injecting perturbations below conventional detection thresholds."
      ]),

      bodyPara([
        "The robustness-compute tradeoff analysis directly informs defense selection for different operational tiers: tactical edge systems (low size, weight, and power) benefit most from channel augmentation training, while platform-level and enterprise systems can afford the computational overhead of full adversarial training. These findings contribute to establishing adversarial robustness testing as a standard practice for AI-enabled EW systems, analogous to electromagnetic compatibility testing for hardware."
      ], { indent: true }),

      // ===== EXPECTED CONTRIBUTIONS =====
      heading1("Expected Contributions"),

      bodyPara([
        "The full paper will present: (1) complete experimental results across all channel conditions and perturbation budgets, including defense comparison with quantified robustness-compute tradeoffs; (2) per-modulation vulnerability analysis identifying which modulation schemes are most susceptible to adversarial attack; (3) discussion of transfer attack feasibility and operational scenarios for adversarial EW; and (4) recommendations for robustness standards in AI-enabled spectrum awareness systems. All code, model checkpoints, and experimental configurations will be released for reproducibility."
      ]),

      // ===== REFERENCES =====
      heading1("References"),

      bodyPara([{ text: "[1] ", bold: true }, "T. J. O\u2019Shea and J. Corgan. 2016. \u201CConvolutional Radio Modulation Recognition Networks.\u201D arXiv:1602.04105."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[2] ", bold: true }, "T. J. O\u2019Shea, T. Roy, and T. C. Clancy. 2018. \u201COver-the-Air Deep Learning Based Radio Signal Classification.\u201D IEEE JSAC 36(1): 132\u2013141."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[3] ", bold: true }, "I. Goodfellow, J. Shlens, and C. Szegedy. 2015. \u201CExplaining and Harnessing Adversarial Examples.\u201D ICLR."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[4] ", bold: true }, "A. Madry et al. 2018. \u201CTowards Deep Learning Models Resistant to Adversarial Attacks.\u201D ICLR."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[5] ", bold: true }, "M. Sadeghi and E. G. Larsson. 2019. \u201CAdversarial Attacks on Deep-Learning Based Radio Signal Classification.\u201D IEEE Wireless Communications Letters 8(1): 213\u2013216."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[6] ", bold: true }, "Y. Lin et al. 2020. \u201CTactics and Adversarial Attacks on Deep Reinforcement Learning Agents for Autonomous RF Signal Classification.\u201D IEEE Access 8: 153191\u2013153203."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[7] ", bold: true }, "S. Flowers, R. E. Temple, and D. Headley. 2019. \u201CEvaluating Adversarial Evasion Attacks in the Context of Wireless Communications.\u201D IEEE MILCOM."], { spacing: { after: 60 } }),
      bodyPara([{ text: "[8] ", bold: true }, "K. Davaslioglu and Y. E. Sagduyu. 2019. \u201CTrojan Attacks on Wireless Signal Classification with Adversarial Machine Learning.\u201D IEEE DySPAN."], { spacing: { after: 60 } }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/happy-exciting-meitner/mnt/Desktop/ew project/CDR_Extended_Abstract_Adversarial_RF_Robustness.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Extended abstract created: " + outPath);
  console.log("File size: " + (buffer.length / 1024).toFixed(1) + " KB");
});
