const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Header, Footer,
        AlignmentType, PageNumber, BorderStyle, Table, TableRow,
        TableCell, WidthType } = require('docx');

// ============================================================
// CDR Special Issue Proposal: EW & Adversarial AI
// ============================================================

const FONT = "Times New Roman";
const TITLE_SIZE = 32; // 16pt
const H1_SIZE = 26; // 13pt
const H2_SIZE = 24; // 12pt
const BODY_SIZE = 22; // 11pt
const SMALL_SIZE = 20; // 10pt

// Placeholder styling
const PLACEHOLDER_COLOR = "CC0000";

function h1(text) {
  return new Paragraph({
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, font: FONT, size: H1_SIZE })]
  });
}

function h2(text) {
  return new Paragraph({
    spacing: { before: 240, after: 100 },
    children: [new TextRun({ text, bold: true, italics: true, font: FONT, size: H2_SIZE })]
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
    ...opts,
    children
  });
}

function placeholder(text) {
  return new TextRun({ text: `[${text}]`, font: FONT, size: BODY_SIZE, color: PLACEHOLDER_COLOR, bold: true });
}

function bullet(textRuns, level = 0) {
  const children = textRuns.map(t => {
    if (typeof t === 'string') return new TextRun({ text: t, font: FONT, size: BODY_SIZE });
    return new TextRun({ font: FONT, size: BODY_SIZE, ...t });
  });
  return new Paragraph({
    spacing: { after: 60, line: 260 },
    indent: { left: 720 + (level * 360), hanging: 360 },
    children: [new TextRun({ text: "\u2022  ", font: FONT, size: BODY_SIZE }), ...children]
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 60 }, children: [] });
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
          children: [
            new TextRun({ text: "Special Issue Proposal", font: FONT, size: SMALL_SIZE, italics: true }),
            new TextRun({ text: "\t\t\t\t\t", font: FONT, size: SMALL_SIZE }),
            new TextRun({ text: "The Cyber Defense Review", font: FONT, size: SMALL_SIZE, italics: true })
          ]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL_SIZE })]
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "UNCLASSIFIED", font: FONT, size: SMALL_SIZE, color: "2E8B57" })]
          })
        ]
      })
    },
    children: [
      // Classification header
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
        children: [new TextRun({ text: "UNCLASSIFIED", font: FONT, size: BODY_SIZE, color: "2E8B57" })]
      }),

      // Title
      new Paragraph({
        spacing: { before: 480, after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Special Issue Proposal", bold: true, font: FONT, size: TITLE_SIZE })]
      }),

      // Subtitle
      new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Adversarial Robustness and AI Security in Electronic Warfare",
          bold: true, italics: true, font: FONT, size: H1_SIZE
        })]
      }),

      new Paragraph({
        spacing: { after: 360 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Submitted to The Cyber Defense Review, Army Cyber Institute", font: FONT, size: BODY_SIZE, italics: true })]
      }),

      // ============================================================
      // SECTION 1: Scope and Focus
      // ============================================================
      h1("1  Scope and Focus of the Special Issue"),
      body([
        { text: "(max. 100 words)", italics: true, color: "999999" }
      ]),
      body([
        "This special issue examines the adversarial robustness of AI and machine learning systems deployed for radio frequency (RF) signal classification in electronic warfare (EW) environments. It addresses a critical question: how vulnerable are deep learning-based spectrum awareness systems to adversarial manipulation, and what defense strategies are viable under the size, weight, and power constraints of tactical edge platforms? The issue brings together perspectives from adversarial machine learning, wireless communications, and cognitive EW to establish adversarial robustness as a first-order system requirement for AI-enabled electromagnetic spectrum operations."
      ]),

      // ============================================================
      // SECTION 2: Rationale and Timeliness
      // ============================================================
      h1("2  Rationale and Timeliness"),
      body([
        { text: "(max. 500 words)", italics: true, color: "999999" }
      ]),

      body([
        "The convergence of three developments makes this special issue both timely and essential. First, the Department of Defense has accelerated integration of artificial intelligence into electromagnetic spectrum operations. Programs such as DARPA\u2019s RF Machine Learning Systems initiative and the DoD Spectrum Access R&D Program explicitly prioritize machine learning for adaptive spectrum sensing, interference mitigation, and cognitive electronic attack. AI-based RF classifiers\u2014trained on in-phase and quadrature (I/Q) baseband samples to identify modulation schemes in real time\u2014now form a foundational capability for next-generation EW systems."
      ]),

      body([
        "Second, the adversarial machine learning community has demonstrated conclusively that deep neural networks are susceptible to adversarial perturbations: small, carefully crafted input modifications that cause misclassification with high confidence (Goodfellow, Shlens, and Szegedy 2015; Madry et al. 2018). While this vulnerability has been extensively characterized in computer vision, its implications for RF signal classification in contested spectrum environments remain underexplored. Recent work has shown that adversarial perturbations can reduce RF classifier accuracy from above 90% to below 10% at perturbation levels invisible to conventional energy detectors (Sadeghi and Larsson 2019; Flowers, Temple, and Headley 2019). This poses a direct threat to cognitive EW capabilities."
      ], { indent: true }),

      body([
        "Third, the physical-layer dimension introduces unique complexities absent from image-domain adversarial research. RF adversarial perturbations must survive wireless channel propagation\u2014including additive noise, multipath fading, and carrier frequency offset\u2014which can attenuate or decorrelate carefully crafted perturbation structures. Conversely, channel-aware adversaries can design perturbations that account for expected channel responses (Lin et al. 2022). This interaction between adversarial manipulation and physical-layer effects represents a largely unexplored frontier with significant operational implications."
      ], { indent: true }),

      body([
        "The proposed special issue addresses this gap by soliciting contributions that span the full adversarial lifecycle: attack characterization under realistic channel conditions, defense mechanisms feasible for edge-deployed systems, certified robustness guarantees adapted for RF domains, and operational frameworks for integrating robustness into EW system design and acquisition. The topic is inherently interdisciplinary, drawing on signal processing, machine learning, cybersecurity, and defense operations\u2014precisely the intersection that CDR\u2019s readership engages."
      ], { indent: true }),

      body([
        "Current geopolitical tensions underscore the urgency. Peer adversaries are investing heavily in electronic attack capabilities, and the electromagnetic spectrum has been explicitly recognized as a contested domain in the 2024 National Defense Strategy. Deploying AI-enabled spectrum awareness systems without systematic adversarial robustness evaluation is analogous to deploying cryptographic systems without adversarial testing\u2014an unacceptable risk. This special issue aims to catalyze the research community\u2019s attention toward establishing adversarial robustness as a standard practice for AI-enabled EW."
      ], { indent: true }),

      // ============================================================
      // SECTION 3: Qualifications of Guest Editors
      // ============================================================
      h1("3  Qualifications of the Guest Editors"),

      body([
        placeholder("RACHEL \u2014 Please fill in your qualifications here. Include:"),
      ]),
      bullet([placeholder("Your name, title, and institutional affiliation")]),
      bullet([placeholder("Degree program (e.g., M.S. Computational Science & Engineering)")]),
      bullet([placeholder("Relevant research experience in RF/ML, EW, or adversarial ML")]),
      bullet([placeholder("Any prior publications, conference presentations, or technical reports")]),
      bullet([placeholder("Professional/military experience related to EW or spectrum operations")]),
      bullet([placeholder("Any co-editors and their qualifications")]),
      emptyLine(),
      body([
        { text: "Note: ", italics: true },
        { text: "You may attach brief CVs or short bios as supplementary material.", italics: true }
      ]),

      // ============================================================
      // SECTION 4: List of Key Potential Contributors
      // ============================================================
      h1("4  List of Key Potential Contributors"),

      body([
        { text: "The following is a preliminary, non-binding list of scholars and practitioners whose expertise aligns with the special issue theme:", italics: false }
      ]),

      emptyLine(),
      body([{ text: "Adversarial Machine Learning for RF/Wireless:", bold: true }]),
      bullet(["Meysam Sadeghi (Link\u00F6ping University) \u2014 adversarial attacks on radio signal classifiers"]),
      bullet(["Yalin E. Sagduyu (Virginia Tech / ARL) \u2014 adversarial ML in wireless networks, trojan attacks"]),
      bullet(["Bryse Flowers (UC San Diego / Navy) \u2014 adversarial evasion in wireless communications"]),
      bullet(["Yi Lin (Southeast University, China) \u2014 channel-aware adversarial attacks on wireless classifiers"]),

      emptyLine(),
      body([{ text: "RF/Modulation Classification and Cognitive EW:", bold: true }]),
      bullet(["Timothy J. O\u2019Shea (DeepSig, Inc.) \u2014 creator of RadioML datasets, deep learning for RF"]),
      bullet(["T. Charles Clancy (MITRE) \u2014 RF machine learning, spectrum security"]),
      bullet(["Karen Z. Haigh (BBN Technologies / Raytheon) \u2014 cognitive electronic warfare"]),

      emptyLine(),
      body([{ text: "Adversarial Robustness and Certified Defenses:", bold: true }]),
      bullet(["Aleksander Madry (MIT) \u2014 PGD adversarial training, robustness evaluation"]),
      bullet(["Jeremy M. Cohen (Carnegie Mellon University) \u2014 certified robustness via randomized smoothing"]),
      bullet(["Francesco Restuccia (Northeastern University) \u2014 over-the-air adversarial deep learning"]),

      emptyLine(),
      body([
        placeholder("RACHEL \u2014 Add any additional contributors from your network, advisors, or program")
      ]),

      // ============================================================
      // SECTION 5: List of Potential Peer Reviewers
      // ============================================================
      h1("5  List of Potential Peer Reviewers"),

      body(["The following researchers have the domain expertise to serve as qualified, independent peer reviewers for submissions in this area:"]),

      emptyLine(),
      bullet(["Salvatore D\u2019Oro, Northeastern University \u2014 wireless adversarial deep learning"]),
      bullet(["Daniel Headley, Virginia Tech \u2014 adversarial evasion in wireless contexts"]),
      bullet(["Kemal Davaslioglu, Virginia Tech / ARL \u2014 adversarial ML for spectrum access"]),
      bullet(["Tamoghna Roy, DeepSig / Virginia Tech \u2014 deep learning-based radio signal classification"]),
      bullet(["Tommaso Melodia, Northeastern University \u2014 wireless networking, RF ML systems"]),
      bullet(["Erik G. Larsson, Link\u00F6ping University \u2014 signal processing, adversarial RF"]),

      emptyLine(),
      body([
        placeholder("RACHEL \u2014 Add 2\u20134 additional reviewers from your academic network, especially faculty advisors or committee members who work in related areas")
      ]),

      emptyLine(),
      body([
        { text: "Note: ", italics: true },
        { text: "None of the above have been contacted or confirmed. This list demonstrates the breadth of qualified expertise available for peer review.", italics: true }
      ]),

      // ============================================================
      // SECTION 6: Draft Call for Papers
      // ============================================================
      h1("6  Draft Call for Papers"),
      body([
        { text: "(max. 1000 words)", italics: true, color: "999999" }
      ]),

      emptyLine(),
      new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "CALL FOR PAPERS", bold: true, font: FONT, size: H1_SIZE })]
      }),
      new Paragraph({
        spacing: { after: 60 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "The Cyber Defense Review \u2014 Special Issue on", font: FONT, size: BODY_SIZE, italics: true })]
      }),
      new Paragraph({
        spacing: { after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Adversarial Robustness and AI Security in Electronic Warfare", bold: true, font: FONT, size: H2_SIZE })]
      }),

      body([
        { text: "Overview. ", bold: true },
        "The Cyber Defense Review invites original research contributions for a special issue dedicated to the adversarial robustness of artificial intelligence and machine learning systems operating in contested electromagnetic spectrum environments. As military forces increasingly depend on AI-enabled spectrum awareness for cognitive electronic warfare, understanding and mitigating adversarial vulnerabilities in these systems has become a critical operational imperative."
      ]),

      body([
        { text: "Motivation. ", bold: true },
        "Deep learning classifiers operating on radio frequency (RF) signals can identify modulation schemes, detect emitters, and support adaptive spectrum management at machine speed. However, these classifiers inherit the adversarial vulnerabilities well-documented in computer vision: small, carefully crafted perturbations can cause high-confidence misclassification. In electronic warfare contexts, a misclassified waveform can trigger incorrect countermeasures, degrade situational awareness, or enable adversary deception. Unlike image-domain attacks, RF adversarial perturbations must contend with wireless channel effects\u2014including noise, fading, and frequency offsets\u2014that may attenuate or amplify their effectiveness. This physical-layer dimension creates unique challenges and opportunities for both attackers and defenders."
      ], { indent: true }),

      body([
        { text: "Scope. ", bold: true },
        "We seek contributions that address the intersection of adversarial machine learning and RF signal processing in defense contexts. Topics of interest include, but are not limited to:"
      ]),

      bullet(["Adversarial attack characterization for RF/modulation classifiers under realistic channel conditions (AWGN, fading, CFO, phase noise)"]),
      bullet(["Channel-aware and over-the-air adversarial attacks on wireless ML systems"]),
      bullet(["Defense mechanisms for edge-deployed EW systems: adversarial training, randomized smoothing, input preprocessing, certified robustness"]),
      bullet(["Robustness-compute tradeoff analysis under size, weight, and power (SWaP) constraints"]),
      bullet(["Transfer attacks, black-box attacks, and query-based attacks in RF domains"]),
      bullet(["Adversarial robustness of emerging architectures (transformers, graph neural networks) for spectrum sensing"]),
      bullet(["Formal verification and certified defenses adapted for physical-layer ML"]),
      bullet(["Operational frameworks for adversarial robustness testing in EW system acquisition and deployment"]),
      bullet(["Red-teaming methodologies for AI-enabled spectrum awareness systems"]),
      bullet(["Policy implications of adversarial AI vulnerability in electromagnetic spectrum operations"]),

      emptyLine(),
      body([
        { text: "Submission Types. ", bold: true },
        "We welcome research articles (6,000\u201310,000 words), shorter technical notes (3,000\u20135,000 words), and perspective pieces (2,000\u20134,000 words). All submissions will undergo double-blind peer review by at least two independent reviewers. Submissions must be original, unpublished, and not under consideration elsewhere."
      ]),

      body([
        { text: "Relevance. ", bold: true },
        "This special issue serves the CDR\u2019s core readership\u2014military cyber professionals, defense researchers, and policymakers\u2014by bridging the gap between theoretical adversarial ML research and the operational realities of contested spectrum environments. Contributors are encouraged to frame findings in terms of operational implications and to consider how adversarial robustness integrates into broader electromagnetic spectrum risk management."
      ], { indent: true }),

      body([
        { text: "Citation Style. ", bold: true },
        "The Cyber Defense Review uses the Chicago Manual of Style, 18th Edition, Author-Date format. In-text citations should use the (Author Year) convention, with a full reference list at the end of the manuscript."
      ], { indent: true }),

      // ============================================================
      // SECTION 7: Schedule and Editorial Plan
      // ============================================================
      h1("7  Schedule and Editorial Plan"),

      body(["The following provisional timeline balances sufficient time for quality peer review with the goal of timely publication aligned with the evolving EW policy landscape:"]),

      emptyLine(),
      bullet([{ text: "Call for Papers publication: ", bold: true }, placeholder("Month Year, e.g., July 2026")]),
      bullet([{ text: "Abstract deadline (optional): ", bold: true }, placeholder("Month Year, e.g., September 2026")]),
      bullet([{ text: "Full submissions due: ", bold: true }, placeholder("Month Year, e.g., December 2026")]),
      bullet([{ text: "Initial review period: ", bold: true }, placeholder("e.g., January\u2013February 2027"), " (3 weeks per reviewer, 2\u20133 reviewers per paper)"]),
      bullet([{ text: "Notification to authors: ", bold: true }, placeholder("Month Year, e.g., March 2027")]),
      bullet([{ text: "Revised versions due: ", bold: true }, placeholder("Month Year, e.g., May 2027"), " (30 days for major revisions, 20 days for minor)"]),
      bullet([{ text: "Final review confirmation: ", bold: true }, placeholder("Month Year, e.g., June 2027"), " (10\u201314 days for revision review)"]),
      bullet([{ text: "Special issue published: ", bold: true }, placeholder("Month Year, e.g., Fall 2027")]),

      emptyLine(),
      body([
        { text: "Review Process. ", bold: true },
        "All submissions will be reviewed by at least two (ideally three) qualified, independent reviewers selected from the pool identified in Section 5 and supplemented as needed. Reviewers will provide written reports shared with authors. Papers submitted by guest editor(s) will be handled under an independent review process coordinated with the Editor-in-Chief. The CFP will be disseminated through relevant academic conferences (IEEE MILCOM, DySPAN, GLOBECOM), professional networks (AFCEA, AOC), and mailing lists in the adversarial ML and wireless security communities."
      ]),

      body([
        { text: "Coordination. ", bold: true },
        "Guest editor(s) will maintain regular communication with the CDR editorial office through biweekly status updates during active review periods. All reviewer assignments, deadline extensions, and editorial decisions will be coordinated with and approved by the Editor-in-Chief."
      ], { indent: true }),

      emptyLine(),
      body([
        { text: "Note: ", italics: true },
        { text: "The final schedule will be determined in coordination with and subject to approval by the journal\u2019s editorial team.", italics: true }
      ]),

      // ============================================================
      // REFERENCES
      // ============================================================
      h1("References"),

      new Paragraph({
        spacing: { after: 80, line: 240 },
        indent: { left: 480, hanging: 480 },
        children: [
          new TextRun({ text: "Flowers, Bryse, R. Michael Temple, and Daniel Headley. 2019. \u201CEvaluating Adversarial Evasion Attacks in the Context of Wireless Communications.\u201D In ", font: FONT, size: REF_SIZE }),
          new TextRun({ text: "Proceedings of the IEEE Military Communications Conference (MILCOM)", font: FONT, size: REF_SIZE, italics: true }),
          new TextRun({ text: ", 1\u20136. Norfolk, VA: IEEE.", font: FONT, size: REF_SIZE })
        ]
      }),
      new Paragraph({
        spacing: { after: 80, line: 240 },
        indent: { left: 480, hanging: 480 },
        children: [
          new TextRun({ text: "Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. 2015. \u201CExplaining and Harnessing Adversarial Examples.\u201D In ", font: FONT, size: REF_SIZE }),
          new TextRun({ text: "Proceedings of the 3rd International Conference on Learning Representations (ICLR)", font: FONT, size: REF_SIZE, italics: true }),
          new TextRun({ text: ". San Diego, CA.", font: FONT, size: REF_SIZE })
        ]
      }),
      new Paragraph({
        spacing: { after: 80, line: 240 },
        indent: { left: 480, hanging: 480 },
        children: [
          new TextRun({ text: "Lin, Yi, Haoyue Zhao, Ya Tu, Songlin Chen, and Guan Gui. 2022. \u201CChannel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers.\u201D ", font: FONT, size: REF_SIZE }),
          new TextRun({ text: "IEEE Transactions on Wireless Communications", font: FONT, size: REF_SIZE, italics: true }),
          new TextRun({ text: " 21 (9): 6951\u20136960.", font: FONT, size: REF_SIZE })
        ]
      }),
      new Paragraph({
        spacing: { after: 80, line: 240 },
        indent: { left: 480, hanging: 480 },
        children: [
          new TextRun({ text: "Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. \u201CTowards Deep Learning Models Resistant to Adversarial Attacks.\u201D In ", font: FONT, size: REF_SIZE }),
          new TextRun({ text: "Proceedings of the 6th International Conference on Learning Representations (ICLR)", font: FONT, size: REF_SIZE, italics: true }),
          new TextRun({ text: ". Vancouver, Canada.", font: FONT, size: REF_SIZE })
        ]
      }),
      new Paragraph({
        spacing: { after: 80, line: 240 },
        indent: { left: 480, hanging: 480 },
        children: [
          new TextRun({ text: "Sadeghi, Meysam, and Erik G. Larsson. 2019. \u201CAdversarial Attacks on Deep-Learning Based Radio Signal Classification.\u201D ", font: FONT, size: REF_SIZE }),
          new TextRun({ text: "IEEE Wireless Communications Letters", font: FONT, size: REF_SIZE, italics: true }),
          new TextRun({ text: " 8 (1): 213\u2013216.", font: FONT, size: REF_SIZE })
        ]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "/sessions/happy-exciting-meitner/mnt/Desktop/ew project/CDR_Special_Issue_Proposal_DRAFT.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Special Issue Proposal created: " + outPath);
  console.log("File size: " + (buffer.length / 1024).toFixed(1) + " KB");
});
