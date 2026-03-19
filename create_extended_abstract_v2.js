const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Header, Footer,
        AlignmentType, HeadingLevel, BorderStyle, PageNumber,
        TabStopType, TabStopPosition } = require('docx');

// ============================================================
// CDR Extended Abstract v2: Chicago 18th Ed Author-Year Citations
// ============================================================

const FONT = "Times New Roman";
const TITLE_SIZE = 28; // 14pt
const HEADING1_SIZE = 24; // 12pt
const BODY_SIZE = 22; // 11pt
const SMALL_SIZE = 20; // 10pt
const REF_SIZE = 20; // 10pt for references

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

function refPara(textRuns) {
  const children = textRuns.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, font: FONT, size: REF_SIZE });
    return new TextRun({ font: FONT, size: REF_SIZE, ...t });
  });
  return new Paragraph({
    spacing: { after: 80, line: 240 },
    alignment: AlignmentType.LEFT,
    indent: { left: 480, hanging: 480 },
    children
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
          children: [new TextRun({ text: "CDR 2026 EW Special Issue \u2014 Extended Abstract", font: FONT, size: SMALL_SIZE, italics: true, color: "666666" })]
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

      // ===== ABSTRACT =====
      new Paragraph({
        spacing: { before: 240, after: 80 },
        children: [new TextRun({ text: "ABSTRACT", bold: true, font: FONT, size: HEADING1_SIZE })]
      }),

      bodyPara([
        { text: "", italics: false },
        "Deep learning classifiers operating on in-phase and quadrature (I/Q) baseband samples have become central to machine-speed spectrum awareness in cognitive electronic warfare (EW). However, their robustness to adversarial perturbations under realistic wireless channel conditions remains poorly characterized. This paper presents a systematic evaluation of adversarial vulnerability for a convolutional neural network (CNN) modulation classifier on the RadioML 2016.10a dataset (O\u2019Shea, Corgan, and Clancy 2016), comprising 11 modulation schemes and 220,000 samples. We define a physically grounded threat model using power-ratio perturbation constraints and evaluate Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks (Goodfellow, Shlens, and Szegedy 2015; Madry et al. 2018) across signal-to-noise ratios (SNR) from \u201310 to 18 dB under three channel conditions: AWGN, Rayleigh fading with AWGN, and composite Rayleigh fading with carrier frequency offset (CFO) and AWGN. Our baseline CNN achieves 70% clean accuracy at high SNR but remains vulnerable to PGD attacks, with attack success rates reaching 24% at 5% perturbation-to-signal power ratio. We compare three lightweight defense strategies\u2014channel augmentation training, adversarial training, and noise injection (Cohen, Rosenfeld, and Kolter 2019)\u2014and analyze their robustness-compute tradeoffs for edge-deployed EW systems. Results demonstrate that channel augmentation provides the most efficient robustness improvement per computational dollar, while adversarial training yields the strongest absolute defense at higher cost."
      ]),

      // Keywords
      new Paragraph({
        spacing: { before: 120, after: 240 },
        children: [
          new TextRun({ text: "Keywords: ", bold: true, font: FONT, size: BODY_SIZE }),
          new TextRun({ text: "adversarial machine learning, RF modulation classification, electronic warfare, adversarial robustness, deep learning, channel impairments", font: FONT, size: BODY_SIZE })
        ]
      }),

      // ===== INTRODUCTION =====
      heading1("Introduction"),

      bodyPara([
        "Artificial intelligence and deep learning have rapidly become essential enablers of machine-speed spectrum awareness in contested electromagnetic environments. Modern radio frequency (RF) classifiers, trained on I/Q baseband samples, can identify modulation schemes and support adaptive spectrum management at speeds far exceeding human operators (O\u2019Shea, Corgan, and Clancy 2016; O\u2019Shea, Roy, and Clancy 2018). These capabilities underpin cognitive electronic warfare (CEW), where real-time waveform classification is a prerequisite for spectrum dominance in multi-domain operations (Haigh et al. 2021)."
      ]),

      bodyPara([
        "However, the broader machine learning community has demonstrated that deep neural networks are susceptible to adversarial perturbations\u2014small, carefully crafted input modifications that cause misclassification with high confidence (Goodfellow, Shlens, and Szegedy 2015). While extensively studied in computer vision (Carlini and Wagner 2017), the implications for physical-layer RF signal classification in contested spectrum environments have received limited attention (Sadeghi and Larsson 2019; Flowers, Temple, and Headley 2019). A misclassified waveform can trigger incorrect countermeasures, degrade situational awareness, or enable adversary deception."
      ], { indent: true }),

      bodyPara([
        "Unlike image classification, RF adversarial perturbations must contend with channel impairments including additive white Gaussian noise, multipath fading, and carrier frequency offset. Prior work has shown that over-the-air channel propagation can degrade digital-domain adversarial perturbations (Restuccia, D\u2019Oro, and Melodia 2020), but recent channel-aware attacks have demonstrated that perturbations can be designed to survive transmission (Lin et al. 2022). Understanding how channel effects interact with adversarial vulnerability is therefore critical for assessing real-world robustness of deployed classifiers."
      ], { indent: true }),

      bodyPara([
        "This paper addresses this gap through a systematic experimental evaluation. We make four contributions: (1) a physically plausible threat model using power-ratio perturbation constraints grounded in RF physics, rather than norm-based bounds borrowed from computer vision; (2) channel-aware robustness evaluation across three channel conditions and SNR levels; (3) comparison of three lightweight defense strategies with operational tradeoff analysis for edge-deployed systems; and (4) framing of findings in the context of cognitive EW and electromagnetic spectrum risk."
      ], { indent: true }),

      // ===== RESEARCH DESIGN =====
      heading1("Research Design"),

      heading2("Dataset and Model"),
      bodyPara([
        "We use the RadioML 2016.10a benchmark dataset (O\u2019Shea, Corgan, and Clancy 2016) containing 220,000 I/Q sample windows across 11 modulation schemes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, 16-QAM, 64-QAM, QPSK, WBFM) spanning SNR levels from \u201320 to +18 dB. We filter to the [\u201310, 18] dB range, yielding 165,000 samples split 70/15/15 for training, validation, and testing. Our baseline classifier is a four-block 1D CNN with 126,475 trainable parameters, inspired by VT-CNN2 (O\u2019Shea, Roy, and Clancy 2018), representative of lightweight architectures suitable for edge deployment."
      ]),

      heading2("Threat Model"),
      bodyPara([
        "We consider a pre-channel adversary who perturbs the transmitted signal before wireless propagation. Rather than adopting the L",
        { text: "\u221E", font: FONT, size: BODY_SIZE },
        " or fixed-epsilon L",
        { text: "\u2082", font: FONT, size: BODY_SIZE },
        " norms common in computer vision (Carlini and Wagner 2017), we impose a power-ratio constraint: the perturbation energy relative to signal energy must satisfy ||\u03B4||",
        { text: "\u2082", font: FONT, size: BODY_SIZE },
        " / ||x||",
        { text: "\u2082", font: FONT, size: BODY_SIZE },
        " \u2264 \u03C1, where \u03C1 ranges from 0.5% to 5% (\u201346 dB to \u201326 dB perturbation-to-signal power ratio). This power-ratio constraint is physically meaningful in RF, directly corresponding to interference-to-signal ratio. We implement FGSM (Goodfellow, Shlens, and Szegedy 2015) and PGD with 10 steps (Madry et al. 2018), both using L",
        { text: "\u2082", font: FONT, size: BODY_SIZE },
        "-normalized gradients adapted for the I/Q domain."
      ]),

      heading2("Channel Conditions"),
      bodyPara([
        "We evaluate under three channel configurations of increasing severity: (1) AWGN only; (2) Rayleigh block fading with AWGN; and (3) composite Rayleigh fading with carrier frequency offset and AWGN. All channel layers are implemented as differentiable PyTorch modules, enabling gradient-based attacks to account for channel effects during perturbation optimization. This approach follows recent work on channel-aware adversarial attacks (Lin et al. 2022) and extends it to a broader set of channel conditions."
      ]),

      heading2("Defense Strategies"),
      bodyPara([
        "We evaluate three defenses chosen for computational feasibility in edge-deployed systems: (1) channel augmentation training, where the model is trained with on-the-fly random channel impairments applied to each batch; (2) adversarial training following Madry et al. (2018), using PGD with 5 inner steps and \u03B1 = 0.5 balancing clean and adversarial loss; and (3) noise injection with optional Monte Carlo smoothing at inference, inspired by randomized smoothing (Cohen, Rosenfeld, and Kolter 2019). We additionally evaluate a combined defense (adversarial training with channel augmentation) to assess whether defense mechanisms compose constructively."
      ]),

      // ===== PRELIMINARY FINDINGS =====
      heading1("Preliminary Findings"),

      bodyPara([
        { text: "Baseline Performance. ", bold: true },
        "The CNN achieves approximately 70% classification accuracy at high SNR (18 dB) and 45.6% at 0 dB, consistent with published benchmarks on RadioML 2016.10a (O\u2019Shea, Roy, and Clancy 2018). Accuracy degrades monotonically with decreasing SNR, dropping below random chance (9.1%) at the lowest levels. Rayleigh fading introduces substantial additional degradation beyond AWGN alone, particularly in the moderate SNR regime (0\u201310 dB), reflecting multiplicative distortion of signal constellation geometry."
      ]),

      bodyPara([
        { text: "Adversarial Vulnerability. ", bold: true },
        "Under PGD attack at \u03C1 = 1% perturbation budget and AWGN channel, the attack success rate reaches approximately 8% at SNR = 10 dB, reducing robust accuracy from 70.1% to 67.6%. At the higher budget of \u03C1 = 5%, the attack success rate increases to 24.4% at SNR = 10 dB, reducing robust accuracy to 53.8%. The gap between clean and robust accuracy is most pronounced at moderate-to-high SNR, where the classifier has sufficient clean accuracy to be meaningfully degraded. At very low SNR, channel noise itself dominates the adversarial perturbation. These findings align with prior observations that RF classifiers are vulnerable to gradient-based attacks at operationally relevant power levels (Sadeghi and Larsson 2019; Flowers, Temple, and Headley 2019)."
      ], { indent: true }),

      bodyPara([
        { text: "Channel Interaction. ", bold: true },
        "A key finding is the interaction between channel impairments and adversarial effectiveness. Because pre-channel perturbations undergo the same fading and noise as the signal, channel effects provide partial attenuation of attack efficacy, consistent with over-the-air degradation reported in prior work (Restuccia, D\u2019Oro, and Melodia 2020). However, this accidental robustness is insufficient as a deliberate defense, as attack success rates remain operationally concerning across all channel conditions."
      ], { indent: true }),

      bodyPara([
        { text: "Defense Tradeoffs. ", bold: true },
        "Preliminary results from defense experiments indicate that channel augmentation training provides the most computationally efficient robustness improvement, requiring no additional computation beyond lightweight channel simulation layers. At SNR = 0 dB, channel augmentation raises clean accuracy from 38.8% (baseline) to 56.7% while reducing attack success rate from 29.8% to 23.5%. Adversarial training yields the strongest absolute robustness improvement\u2014lowering ASR from 29.8% to 25.0% at SNR = 0 dB\u2014but at approximately 3\u20135x longer training time. The combined defense (adversarial training with channel augmentation) achieves the lowest ASR of 22.2% at SNR = 0 dB. These tradeoffs are particularly relevant for edge-deployed EW systems operating under size, weight, and power (SWaP) constraints."
      ], { indent: true }),

      // ===== IMPLICATIONS =====
      heading1("Implications for Cyber Defense"),

      bodyPara([
        "Our findings have direct implications for the design and deployment of AI-based spectrum awareness systems in contested electromagnetic environments. The demonstrated vulnerability of baseline RF classifiers to low-power adversarial perturbations indicates that adversarial robustness must be treated as a first-order system requirement, not a theoretical concern (Davaslioglu and Sagduyu 2019). An unprotected spectrum awareness system would be susceptible to manipulation by adversaries injecting perturbations below conventional detection thresholds."
      ]),

      bodyPara([
        "The robustness-compute tradeoff analysis directly informs defense selection for different operational tiers: tactical edge systems (low SWaP) benefit most from channel augmentation training, while platform-level and enterprise systems can afford the computational overhead of full adversarial training. These findings contribute to establishing adversarial robustness testing as a standard practice for AI-enabled EW systems, analogous to electromagnetic compatibility testing for hardware. As the Department of Defense accelerates integration of AI/ML into spectrum operations (Haigh et al. 2021), the need for systematic robustness evaluation frameworks will only intensify."
      ], { indent: true }),

      // ===== EXPECTED CONTRIBUTIONS =====
      heading1("Expected Contributions"),

      bodyPara([
        "The full paper will present: (1) complete experimental results across all channel conditions and perturbation budgets, including per-modulation vulnerability analysis identifying which modulation schemes are most susceptible to adversarial attack; (2) quantified defense comparison with robustness-compute tradeoff curves for each defense strategy; (3) discussion of transfer attack feasibility and operational scenarios for adversarial EW; and (4) recommendations for robustness standards in AI-enabled spectrum awareness systems. All code, model checkpoints, and experimental configurations will be released for reproducibility."
      ]),

      // ===== REFERENCES =====
      heading1("References"),

      // Carlini and Wagner 2017
      refPara([
        "Carlini, Nicholas, and David Wagner. 2017. \u201CTowards Evaluating the Robustness of Neural Networks.\u201D In ",
        { text: "Proceedings of the 2017 IEEE Symposium on Security and Privacy (SP)", italics: true },
        ", 39\u201357. San Jose, CA: IEEE. https://doi.org/10.1109/SP.2017.49."
      ]),

      // Cohen, Rosenfeld, and Kolter 2019
      refPara([
        "Cohen, Jeremy M., Elan Rosenfeld, and J. Zico Kolter. 2019. \u201CCertified Adversarial Robustness via Randomized Smoothing.\u201D In ",
        { text: "Proceedings of the 36th International Conference on Machine Learning (ICML)", italics: true },
        ", 1310\u20131320. Long Beach, CA: PMLR."
      ]),

      // Davaslioglu and Sagduyu 2019
      refPara([
        "Davaslioglu, Kemal, and Yalin E. Sagduyu. 2019. \u201CTrojan Attacks on Wireless Signal Classification with Adversarial Machine Learning.\u201D In ",
        { text: "Proceedings of the IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN)", italics: true },
        ", 1\u20136. Newark, NJ: IEEE. https://doi.org/10.1109/DySPAN.2019.8935684."
      ]),

      // Flowers, Temple, and Headley 2019
      refPara([
        "Flowers, Bryse, R. Michael Temple, and Daniel Headley. 2019. \u201CEvaluating Adversarial Evasion Attacks in the Context of Wireless Communications.\u201D In ",
        { text: "Proceedings of the IEEE Military Communications Conference (MILCOM)", italics: true },
        ", 1\u20136. Norfolk, VA: IEEE. https://doi.org/10.1109/MILCOM47813.2019.9020716."
      ]),

      // Goodfellow, Shlens, and Szegedy 2015
      refPara([
        "Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. 2015. \u201CExplaining and Harnessing Adversarial Examples.\u201D In ",
        { text: "Proceedings of the 3rd International Conference on Learning Representations (ICLR)", italics: true },
        ". San Diego, CA."
      ]),

      // Haigh et al. 2021
      refPara([
        "Haigh, Karen Z., Julia Andrusenko, Luke Brueggeman, and Ryan Hillstrom. 2021. \u201CCognitive Electronic Warfare: Radio Frequency Spectrum Meets Machine Learning.\u201D ",
        { text: "IEEE Communications Magazine", italics: true },
        " 59 (11): 44\u201350. https://doi.org/10.1109/MCOM.001.2100116."
      ]),

      // Lin et al. 2022
      refPara([
        "Lin, Yi, Haoyue Zhao, Ya Tu, Songlin Chen, and Guan Gui. 2022. \u201CChannel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers.\u201D ",
        { text: "IEEE Transactions on Wireless Communications", italics: true },
        " 21 (9): 6951\u20136960. https://doi.org/10.1109/TWC.2022.3153737."
      ]),

      // Madry et al. 2018
      refPara([
        "Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. \u201CTowards Deep Learning Models Resistant to Adversarial Attacks.\u201D In ",
        { text: "Proceedings of the 6th International Conference on Learning Representations (ICLR)", italics: true },
        ". Vancouver, Canada."
      ]),

      // O'Shea, Corgan, and Clancy 2016
      refPara([
        "O\u2019Shea, Timothy J., Johnathan Corgan, and T. Charles Clancy. 2016. \u201CConvolutional Radio Modulation Recognition Networks.\u201D In ",
        { text: "Proceedings of the International Conference on Engineering Applications of Neural Networks", italics: true },
        ", 213\u2013226. Cham: Springer. https://doi.org/10.1007/978-3-319-44188-7_16."
      ]),

      // O'Shea, Roy, and Clancy 2018
      refPara([
        "O\u2019Shea, Timothy J., Tamoghna Roy, and T. Charles Clancy. 2018. \u201COver-the-Air Deep Learning Based Radio Signal Classification.\u201D ",
        { text: "IEEE Journal of Selected Topics in Signal Processing", italics: true },
        " 12 (1): 168\u2013179. https://doi.org/10.1109/JSTSP.2018.2797022."
      ]),

      // Restuccia, D'Oro, and Melodia 2020
      refPara([
        "Restuccia, Francesco, Salvatore D\u2019Oro, and Tommaso Melodia. 2020. \u201CGeneralized Wireless Adversarial Deep Learning.\u201D In ",
        { text: "Proceedings of the 3rd ACM Workshop on Wireless Security and Machine Learning (WiseML)", italics: true },
        ", 1\u20136. Linz, Austria: ACM. https://doi.org/10.1145/3395352.3402622."
      ]),

      // Sadeghi and Larsson 2019
      refPara([
        "Sadeghi, Meysam, and Erik G. Larsson. 2019. \u201CAdversarial Attacks on Deep-Learning Based Radio Signal Classification.\u201D ",
        { text: "IEEE Wireless Communications Letters", italics: true },
        " 8 (1): 213\u2013216. https://doi.org/10.1109/LWC.2018.2867459."
      ]),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/happy-exciting-meitner/mnt/Desktop/ew project/CDR_Extended_Abstract_Adversarial_RF_Robustness.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Extended abstract v2 created: " + outPath);
  console.log("File size: " + (buffer.length / 1024).toFixed(1) + " KB");
});
