const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Header, Footer,
        AlignmentType, PageNumber, BorderStyle } = require('docx');

// ============================================================
// CDR 500-Word Extended Abstract (Early Feedback Submission)
// ============================================================

const FONT = "Times New Roman";
const TITLE_SIZE = 28;
const H1_SIZE = 24;
const BODY_SIZE = 22;
const SMALL_SIZE = 20;
const REF_SIZE = 20;

function h1(text) {
  return new Paragraph({
    spacing: { before: 280, after: 100 },
    children: [new TextRun({ text, bold: true, font: FONT, size: H1_SIZE })]
  });
}

function body(textRuns, opts = {}) {
  const children = textRuns.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, font: FONT, size: BODY_SIZE });
    return new TextRun({ font: FONT, size: BODY_SIZE, ...t });
  });
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.indent ? { firstLine: 360 } : undefined,
    children
  });
}

function refPara(textRuns) {
  const children = textRuns.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, font: FONT, size: REF_SIZE });
    return new TextRun({ font: FONT, size: REF_SIZE, ...t });
  });
  return new Paragraph({
    spacing: { after: 60, line: 240 },
    alignment: AlignmentType.LEFT,
    indent: { left: 480, hanging: 480 },
    children
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: BODY_SIZE } } }
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
          children: [new TextRun({ text: "CDR 2026 EW Special Issue \u2014 500-Word Extended Abstract", font: FONT, size: SMALL_SIZE, italics: true, color: "666666" })]
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
        spacing: { after: 80 },
        children: [new TextRun({ text: "RESEARCH ARTICLE \u2014 EXTENDED ABSTRACT", bold: true, font: FONT, size: BODY_SIZE })]
      }),

      // Title
      new Paragraph({
        spacing: { before: 200, after: 200 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Adversarial Robustness of AI-Based RF Signal Classification Under Realistic Channel Impairments",
          bold: true, font: FONT, size: TITLE_SIZE
        })]
      }),

      // Author placeholder
      new Paragraph({
        spacing: { after: 80 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Author(s): Removed for double-blind review", font: FONT, size: BODY_SIZE, italics: true, color: "888888" })]
      }),
      new Paragraph({
        spacing: { after: 200 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Affiliation(s): Removed for double-blind review", font: FONT, size: BODY_SIZE, italics: true, color: "888888" })]
      }),

      // Keywords
      new Paragraph({
        spacing: { after: 200 },
        children: [
          new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: BODY_SIZE }),
          new TextRun({ text: "adversarial machine learning, RF modulation classification, electronic warfare, adversarial robustness, channel impairments", font: FONT, size: BODY_SIZE })
        ]
      }),

      // Divider
      new Paragraph({
        spacing: { after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } },
        children: []
      }),

      // === BODY ===
      // Motivation (~100 words)
      h1("Motivation"),

      body([
        "Deep learning classifiers operating on in-phase and quadrature (I/Q) baseband samples have become essential for machine-speed spectrum awareness in cognitive electronic warfare (Haigh et al. 2021). However, adversarial perturbations\u2014small, carefully crafted input modifications\u2014can cause high-confidence misclassification in neural networks (Goodfellow, Shlens, and Szegedy 2015). In electronic warfare contexts, a misclassified waveform can trigger incorrect countermeasures or enable adversary deception. While adversarial vulnerability has been demonstrated for RF classifiers in idealized settings (Sadeghi and Larsson 2019), the interaction between adversarial attacks and realistic wireless channel impairments\u2014including fading, noise, and frequency offsets\u2014remains poorly characterized."
      ]),

      // Approach (~150 words)
      h1("Approach"),

      body([
        "We conduct a systematic evaluation of adversarial robustness for a CNN-based modulation classifier on the RadioML 2016.10a dataset (O\u2019Shea, Corgan, and Clancy 2016), comprising 11 modulation schemes and 220,000 I/Q samples. We define a physically grounded threat model using power-ratio perturbation constraints (\u03C1 = ||\u03B4||",
        { text: "\u2082", font: FONT },
        " / ||x||",
        { text: "\u2082", font: FONT },
        "), ranging from 0.5% to 5% perturbation-to-signal power ratio, and evaluate FGSM and PGD attacks (Madry et al. 2018) across SNR levels from \u221210 to 18 dB under three channel conditions: AWGN, Rayleigh fading with AWGN, and composite fading with carrier frequency offset. We compare three lightweight defenses\u2014channel augmentation training, adversarial training, and noise injection (Cohen, Rosenfeld, and Kolter 2019)\u2014and analyze robustness-compute tradeoffs relevant to edge-deployed EW systems operating under size, weight, and power constraints."
      ]),

      // Preliminary Results (~120 words)
      h1("Preliminary Results"),

      body([
        "Our baseline CNN achieves 70% classification accuracy at 18 dB SNR. Under PGD attack at \u03C1 = 1%, attack success rates reach 8% at SNR = 10 dB, increasing to 24% at \u03C1 = 5%. Channel propagation partially attenuates pre-channel perturbations but does not eliminate the threat. Among defenses, channel augmentation training provides the most computationally efficient robustness improvement, raising clean accuracy at SNR = 0 dB from 39% to 54% while reducing attack success rate from 29% to 16%. Adversarial training with channel augmentation achieves the lowest attack success rate (22%) but at 3\u20135x longer training time. These tradeoffs directly inform defense selection across tactical, platform, and enterprise operational tiers."
      ]),

      // Significance (~80 words)
      h1("Significance for Cyber Defense"),

      body([
        "This work establishes that adversarial robustness must be treated as a first-order requirement for AI-enabled spectrum awareness systems, not a theoretical concern. The demonstrated vulnerability at sub-detection-threshold perturbation levels highlights a concrete electromagnetic spectrum risk. Our robustness-compute tradeoff framework directly supports defense selection for EW system designers. As DoD accelerates AI integration into spectrum operations, systematic adversarial robustness evaluation should become standard practice\u2014analogous to electromagnetic compatibility testing for hardware."
      ]),

      // Divider before references
      new Paragraph({
        spacing: { before: 200, after: 120 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 1 } },
        children: []
      }),

      // === REFERENCES ===
      h1("References"),

      refPara([
        "Cohen, Jeremy M., Elan Rosenfeld, and J. Zico Kolter. 2019. \u201CCertified Adversarial Robustness via Randomized Smoothing.\u201D In ",
        { text: "Proceedings of the 36th International Conference on Machine Learning (ICML)", italics: true },
        ", 1310\u20131320. Long Beach, CA: PMLR."
      ]),

      refPara([
        "Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. 2015. \u201CExplaining and Harnessing Adversarial Examples.\u201D In ",
        { text: "Proceedings of the 3rd International Conference on Learning Representations (ICLR)", italics: true },
        ". San Diego, CA."
      ]),

      refPara([
        "Haigh, Karen Z., Julia Andrusenko, Luke Brueggeman, and Ryan Hillstrom. 2021. \u201CCognitive Electronic Warfare: Radio Frequency Spectrum Meets Machine Learning.\u201D ",
        { text: "IEEE Communications Magazine", italics: true },
        " 59 (11): 44\u201350."
      ]),

      refPara([
        "Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. \u201CTowards Deep Learning Models Resistant to Adversarial Attacks.\u201D In ",
        { text: "Proceedings of the 6th International Conference on Learning Representations (ICLR)", italics: true },
        ". Vancouver, Canada."
      ]),

      refPara([
        "O\u2019Shea, Timothy J., Johnathan Corgan, and T. Charles Clancy. 2016. \u201CConvolutional Radio Modulation Recognition Networks.\u201D In ",
        { text: "Proceedings of the International Conference on Engineering Applications of Neural Networks", italics: true },
        ", 213\u2013226. Cham: Springer."
      ]),

      refPara([
        "Sadeghi, Meysam, and Erik G. Larsson. 2019. \u201CAdversarial Attacks on Deep-Learning Based Radio Signal Classification.\u201D ",
        { text: "IEEE Wireless Communications Letters", italics: true },
        " 8 (1): 213\u2013216."
      ]),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/happy-exciting-meitner/mnt/Desktop/ew project/CDR_500Word_Extended_Abstract.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("500-word abstract created: " + outPath);
  console.log("File size: " + (buffer.length / 1024).toFixed(1) + " KB");
});
