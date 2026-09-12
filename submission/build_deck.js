// Builds the SETU pitch deck for the Snapdragon AI Lab Build & Present Challenge.
// Palette is taken from the product UI itself, so the deck and the app read as one thing.

const pptxgen = require("pptxgenjs");

const INK = "0D1420";      // the app's compute strip
const TEAL = "0F7B6C";     // the app's accent
const TEAL_SOFT = "E6F2F0";
const RED = "B3261E";      // the app's danger colour
const RED_SOFT = "FDECEA";
const PAPER = "FFFFFF";
const MUTED = "5B6672";
const LINE = "DFE4EA";

const H = "Cambria";
const B = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "SETU";
pres.title = "SETU - offline comprehension engine";

const W = 13.3;
const M = 0.7; // margin

// ---------------------------------------------------------------- helpers

function titleSlide() {
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("SETU", {
    x: M, y: 1.5, w: 8, h: 1.2,
    fontFace: H, fontSize: 72, bold: true, color: PAPER, isTextBox: true, margin: 0,
  });
  s.addText("सेतु  ·  the bridge", {
    x: M, y: 2.7, w: 8, h: 0.5,
    fontFace: B, fontSize: 18, color: "8EA0B8", isTextBox: true, margin: 0,
  });

  s.addText("An offline comprehension engine for India's language gap.", {
    x: M, y: 3.6, w: 9.6, h: 0.6,
    fontFace: H, fontSize: 28, color: PAPER, isTextBox: true, margin: 0,
  });
  s.addText(
    "It does not translate a conversation. It checks that the person understood it — " +
    "entirely on the laptop, with the network switched off.",
    {
      x: M, y: 4.35, w: 9.6, h: 0.9,
      fontFace: B, fontSize: 16, color: "CFD8E3", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    }
  );

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 5.9, w: 4.6, h: 0.55, fill: { color: "182437" }, rectRadius: 0.1,
  });
  s.addText("Snapdragon® AI Lab Build & Present Challenge", {
    x: M, y: 5.9, w: 4.6, h: 0.55,
    fontFace: B, fontSize: 12, color: "6EE7B7", align: "center", valign: "middle", isTextBox: true,
  });
  s.addNotes(
    "SETU means bridge. The one-line claim: anyone can build a translator; SETU verifies " +
    "comprehension. Everything runs on the device."
  );
  return s;
}

function problemSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("The problem nobody measures", {
    x: M, y: 0.55, w: 8.5, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });

  const lines = [
    { text: "A patient is told, in English, that they have hypertension. ", options: { color: INK } },
    { text: "Take amlodipine 5 mg od. Get a fasting lipid profile. Come back immediately if you get chest pain.", options: { color: INK, bold: true } },
  ];
  s.addText(lines, {
    x: M, y: 1.6, w: 7.2, h: 1.3,
    fontFace: B, fontSize: 17, isTextBox: true, margin: 0, lineSpacingMultiple: 1.35,
  });

  s.addText("They nod. They say “haan ji.” They go home.", {
    x: M, y: 3.0, w: 7.2, h: 0.45,
    fontFace: H, fontSize: 21, italic: true, color: TEAL, isTextBox: true, margin: 0,
  });

  s.addText(
    "They did not understand “od”. They did not understand “fasting”. And they do not " +
    "know that chest pain means return now, not at the follow-up.\n\n" +
    "This is not a translation problem — a phone can translate. It is a comprehension " +
    "problem, and its defining feature is that nobody in the room ever finds out.",
    {
      x: M, y: 3.7, w: 7.2, h: 2.2,
      fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    }
  );

  // stat block
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.4, y: 1.6, w: 4.2, h: 2.0, fill: { color: TEAL_SOFT }, rectRadius: 0.12,
  });
  s.addText("~50%", {
    x: 8.4, y: 1.75, w: 4.2, h: 1.0,
    fontFace: H, fontSize: 60, bold: true, color: TEAL, align: "center", isTextBox: true, margin: 0,
  });
  s.addText("adherence to long-term therapy\nin developing countries (WHO)", {
    x: 8.6, y: 2.75, w: 3.8, h: 0.7,
    fontFace: B, fontSize: 12, color: TEAL, align: "center", isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 8.4, y: 3.85, w: 4.2, h: 2.0, fill: { color: "F4F6F8" }, rectRadius: 0.12,
  });
  s.addText("1 billion+", {
    x: 8.4, y: 4.0, w: 4.2, h: 1.0,
    fontFace: H, fontSize: 46, bold: true, color: INK, align: "center", isTextBox: true, margin: 0,
  });
  s.addText("Indians who do not speak the language\ntheir own paperwork is written in", {
    x: 8.6, y: 4.95, w: 3.8, h: 0.7,
    fontFace: B, fontSize: 12, color: MUTED, align: "center", isTextBox: true, margin: 0,
  });

  s.addNotes("The stakes are ordinary and enormous: the instruction was given and never landed.");
}

function solutionSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("What SETU does", {
    x: M, y: 0.5, w: 8, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText("A laptop on the desk listens to the whole visit. Wi-Fi off.", {
    x: M, y: 1.2, w: 9, h: 0.4,
    fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0,
  });

  const steps = [
    ["1", "Listen", "Voice activity gates the mic so only real speech reaches Whisper. Continuous, all day, on battery."],
    ["2", "Bridge", "Every turn transcribed, language-identified, shown in the patient's own language."],
    ["3", "Flag jargon", "“od”, “fasting”, “lipid profile” — caught as spoken, glossed in plain words."],
    ["4", "Build the plan", "Instructions extracted as they are given: medicines, tests, follow-up, red flags."],
    ["5", "Teach back", "The patient repeats the plan. SETU checks it against what was actually said."],
  ];

  let y = 1.85;
  for (const [n, head, body] of steps) {
    s.addShape(pres.ShapeType.ellipse, {
      x: M, y: y, w: 0.5, h: 0.5, fill: { color: TEAL },
    });
    s.addText(n, {
      x: M, y: y, w: 0.5, h: 0.5,
      fontFace: H, fontSize: 18, bold: true, color: PAPER, align: "center", valign: "middle", isTextBox: true,
    });
    s.addText(head, {
      x: 1.4, y: y - 0.04, w: 2.4, h: 0.35,
      fontFace: H, fontSize: 18, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(body, {
      x: 3.8, y: y - 0.02, w: 8.8, h: 0.6,
      fontFace: B, fontSize: 14, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
    });
    y += 0.95;
  }
  s.addNotes("Step 5 is the product. The first four are how we earn the right to do it.");
}

function momentSlide() {
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("The moment that matters", {
    x: M, y: 0.55, w: 9, h: 0.6,
    fontFace: H, fontSize: 36, bold: true, color: PAPER, isTextBox: true, margin: 0,
  });
  s.addText("The doctor asks the patient to say the plan back. SETU scores it.", {
    x: M, y: 1.2, w: 10, h: 0.4,
    fontFace: B, fontSize: 15, color: "8EA0B8", isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 1.95, w: 11.9, h: 1.5, fill: { color: RED_SOFT }, rectRadius: 0.12,
  });
  s.addText("2 of 5 understood", {
    x: 1.0, y: 2.15, w: 6, h: 0.6,
    fontFace: H, fontSize: 34, bold: true, color: RED, isTextBox: true, margin: 0,
  });
  s.addText("The patient did not repeat these back — and they are the ones that matter.", {
    x: 1.0, y: 2.75, w: 10.5, h: 0.4,
    fontFace: B, fontSize: 14, color: RED, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 3.65, w: 11.9, h: 0.85, fill: { color: "182437" }, rectRadius: 0.1,
  });
  s.addText(
    "✗   “If you get chest pain or breathlessness, come immediately to the emergency.”",
    {
      x: 1.0, y: 3.65, w: 11.2, h: 0.85,
      fontFace: B, fontSize: 16, color: "FF8A80", valign: "middle", isTextBox: true, margin: 0,
    }
  );

  s.addText(
    "The doctor learns this while the patient is still in the room — and the patient leaves " +
    "with a printed card, in their own language, ordered by what will hurt them if they forget it.",
    {
      x: M, y: 4.8, w: 11.9, h: 0.9,
      fontFace: B, fontSize: 15, color: "CFD8E3", isTextBox: true, margin: 0, lineSpacingMultiple: 1.3,
    }
  );

  s.addText("Anyone can build a translator. This is a clinical safety instrument.", {
    x: M, y: 5.95, w: 11.9, h: 0.5,
    fontFace: H, fontSize: 20, italic: true, color: "6EE7B7", isTextBox: true, margin: 0,
  });
  s.addNotes("This is the demo beat. Hold on it.");
}

function domainsSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("One engine, three settings", {
    x: M, y: 0.5, w: 9, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText(
    "Strip the vocabulary away and these are the same problem: two people, a knowledge gap, " +
    "one side issuing instructions — and nobody checking they landed.",
    {
      x: M, y: 1.2, w: 11.9, h: 0.7,
      fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25,
    }
  );

  const cards = [
    ["Clinic", "doctor → patient", "od  ·  fasting  ·  lipid profile", "Red flags first", TEAL],
    ["Classroom", "teacher → student", "weightage  ·  internal  ·  plagiarism", "Exam dates first", "1C7293"],
    ["Counter", "officer → citizen", "self attested  ·  BPL  ·  acknowledgement", "Deadline first", "6D2E46"],
  ];

  let x = M;
  for (const [name, who, jargon, order, colour] of cards) {
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 2.2, w: 3.8, h: 2.9, fill: { color: "F4F6F8" }, rectRadius: 0.12,
    });
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.35, y: 2.55, w: 0.42, h: 0.42, fill: { color: colour } });
    s.addText(name, {
      x: x + 0.95, y: 2.5, w: 2.6, h: 0.5,
      fontFace: H, fontSize: 22, bold: true, color: INK, valign: "middle", isTextBox: true, margin: 0,
    });
    s.addText(who, {
      x: x + 0.35, y: 3.12, w: 3.1, h: 0.3,
      fontFace: B, fontSize: 13, color: colour, bold: true, isTextBox: true, margin: 0,
    });
    s.addText("Catches", {
      x: x + 0.35, y: 3.55, w: 3.1, h: 0.25,
      fontFace: B, fontSize: 10, color: MUTED, isTextBox: true, margin: 0,
    });
    s.addText(jargon, {
      x: x + 0.35, y: 3.78, w: 3.1, h: 0.6,
      fontFace: B, fontSize: 12, color: INK, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
    });
    s.addText("Card ordered by", {
      x: x + 0.35, y: 4.42, w: 3.1, h: 0.25,
      fontFace: B, fontSize: 10, color: MUTED, isTextBox: true, margin: 0,
    });
    s.addText(order, {
      x: x + 0.35, y: 4.64, w: 3.1, h: 0.3,
      fontFace: B, fontSize: 12, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    x += 4.05;
  }

  s.addText(
    "Adding a courtroom or a bank means writing one Domain object — a lexicon, categories, " +
    "questions, headings. Not another app. The test suite fails if one domain's vocabulary leaks into another.",
    {
      x: M, y: 5.4, w: 11.9, h: 0.8,
      fontFace: B, fontSize: 14, color: MUTED, italic: true, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25,
    }
  );
  s.addNotes("Breadth without shallowness: same engine, tested separation.");
}

function snapdragonSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("Why this needs Snapdragon", {
    x: M, y: 0.5, w: 9, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText("Only the third of these is a performance argument.", {
    x: M, y: 1.2, w: 9, h: 0.4,
    fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0,
  });

  const reasons = [
    ["Un-cloudable by law", "A consultation is protected health information. “Send the audio to a data centre” is not a design choice we get to make. On-device is the requirement, not the optimisation."],
    ["Works where the link does not", "A primary health centre with an intermittent connection is the normal case, not the edge case."],
    ["Sustained, not bursty", "Continuous speech recognition across a six-hour clinic day is precisely what an NPU exists for and what a CPU cannot afford. Measurable — so we measure it."],
  ];

  let y = 1.9;
  for (const [head, body] of reasons) {
    s.addShape(pres.ShapeType.roundRect, {
      x: M, y: y, w: 11.9, h: 1.35, fill: { color: "F4F6F8" }, rectRadius: 0.12,
    });
    s.addText(head, {
      x: 1.1, y: y + 0.16, w: 5.0, h: 0.4,
      fontFace: H, fontSize: 20, bold: true, color: TEAL, isTextBox: true, margin: 0,
    });
    s.addText(body, {
      x: 1.1, y: y + 0.58, w: 10.8, h: 0.7,
      fontFace: B, fontSize: 13.5, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.2,
    });
    y += 1.5;
  }
  s.addNotes("Lead with the legal argument; the performance argument lands harder after it.");
}

function routerSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("Hexa-Router", {
    x: M, y: 0.5, w: 9, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText(
    "Most on-device demos hard-code “put everything on the NPU”. That is wrong for a real " +
    "workload — the NPU is shared and finite, and the right answer changes the moment the charger comes out.",
    {
      x: M, y: 1.2, w: 11.9, h: 0.75,
      fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25,
    }
  );

  const inputs = [
    ["Deployment envelope", "What each model can run on at all"],
    ["Live power state", "AC vs battery, charge level, thermal pressure"],
    ["Request priority", "Interactive · streaming · background"],
    ["Learned cost model", "Measured latency and energy, on THIS machine"],
  ];

  let x = M;
  for (const [head, body] of inputs) {
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 2.2, w: 2.9, h: 1.5, fill: { color: TEAL_SOFT }, rectRadius: 0.1,
    });
    s.addText(head, {
      x: x + 0.22, y: 2.4, w: 2.5, h: 0.55,
      fontFace: H, fontSize: 14.5, bold: true, color: TEAL, isTextBox: true, margin: 0,
    });
    s.addText(body, {
      x: x + 0.22, y: 2.95, w: 2.5, h: 0.65,
      fontFace: B, fontSize: 11.5, color: TEAL, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
    });
    x += 3.05;
  }

  s.addText("Decides placement per model, per request — NPU / GPU / CPU", {
    x: M, y: 3.95, w: 11.9, h: 0.4,
    fontFace: H, fontSize: 17, bold: true, color: INK, align: "center", isTextBox: true, margin: 0,
  });

  const facts = [
    ["Hysteresis", "Rebuilding a QNN session costs hundreds of ms, so a candidate must beat the incumbent by a margin before it is worth paying for."],
    ["Thermal backstop", "Work is steered off a throttling NPU before it costs more than it saves."],
    ["HTP performance mode", "burst for interactive turns, sustained_high_performance for all-day streaming, power_saver under battery pressure."],
  ];
  let y = 4.5;
  for (const [head, body] of facts) {
    s.addText(head, {
      x: M, y: y, w: 2.6, h: 0.3,
      fontFace: B, fontSize: 13, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(body, {
      x: 3.4, y: y, w: 9.2, h: 0.5,
      fontFace: B, fontSize: 12.5, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
    });
    y += 0.62;
  }
  s.addNotes("The learned cost model is the part that is genuinely unusual.");
}

function proofSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("Does it actually work?", {
    x: M, y: 0.5, w: 9, h: 0.7,
    fontFace: H, fontSize: 40, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText(
    "Most submissions say “we built it”. We measured it, on a labelled corpus of sessions " +
    "across all three settings.",
    {
      x: M, y: 1.2, w: 11.9, h: 0.4,
      fontFace: B, fontSize: 15, color: MUTED, isTextBox: true, margin: 0,
    }
  );

  const stats = [
    ["0", "dangerous misses", "Told the doctor the patient understood when they had not", RED],
    ["100%", "recall on missed", "Of instructions truly not restated, how many we caught", TEAL],
    ["86.7%", "precision", "Of what we flagged, how much was genuinely missed", TEAL],
    ["99", "tests passing", "Runs green on a bare checkout with no models present", INK],
  ];

  let x = M;
  for (const [big, label, sub, colour] of stats) {
    s.addShape(pres.ShapeType.roundRect, {
      x: x, y: 1.95, w: 2.9, h: 2.15, fill: { color: "F4F6F8" }, rectRadius: 0.12,
    });
    s.addText(big, {
      x: x, y: 2.15, w: 2.9, h: 0.85,
      fontFace: H, fontSize: 44, bold: true, color: colour, align: "center", isTextBox: true, margin: 0,
    });
    s.addText(label, {
      x: x, y: 2.98, w: 2.9, h: 0.3,
      fontFace: B, fontSize: 13, bold: true, color: INK, align: "center", isTextBox: true, margin: 0,
    });
    s.addText(sub, {
      x: x + 0.2, y: 3.3, w: 2.5, h: 0.7,
      fontFace: B, fontSize: 10.5, color: MUTED, align: "center", isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
    });
    x += 3.05;
  }

  s.addText("The evaluation overturned our own design", {
    x: M, y: 4.4, w: 11.9, h: 0.4,
    fontFace: H, fontSize: 20, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText(
    "Replacing keyword matching with sentence embeddings made the check WORSE — precision fell " +
    "81% → 76%. Cosine scores an imperative against a first-person promise lower than expected, " +
    "and the covered/missed ranges overlap. The two signals failed on different items, so the " +
    "shipped scorer unions them: 86.7%, better than either alone.\n\n" +
    "Thresholds were chosen at the operating point that keeps dangerous misses at zero — not " +
    "the one that maximises F1. A false alarm costs a doctor five seconds. A dangerous miss costs a patient.",
    {
      x: M, y: 4.85, w: 11.9, h: 1.9,
      fontFace: B, fontSize: 13, color: MUTED, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25,
    }
  );
  s.addNotes("An eval that contradicted our design is the hardest thing for a rival entry to fake.");
}

function statusSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };

  s.addText("What runs today — and what does not", {
    x: M, y: 0.5, w: 11, h: 0.7,
    fontFace: H, fontSize: 38, bold: true, color: INK, isTextBox: true, margin: 0,
  });

  s.addText("Real inference, verified", {
    x: M, y: 1.35, w: 5.6, h: 0.4,
    fontFace: H, fontSize: 18, bold: true, color: TEAL, isTextBox: true, margin: 0,
  });
  const done = [
    "Whisper speech recognition — verified on real audio",
    "Live microphone → transcript → care plan",
    "Translation into 8 Indian languages",
    "Sentence embeddings, voice activity detection",
    "Document scanning — prescription to care plan",
    "Hexa-Router with measured battery draw",
  ];
  s.addText(done.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < done.length - 1 } })), {
    x: M, y: 1.8, w: 5.6, h: 2.4,
    fontFace: B, fontSize: 13, color: INK, isTextBox: true, margin: 0, paraSpaceAfter: 8,
  });

  s.addText("Honestly not finished", {
    x: 6.9, y: 1.35, w: 5.7, h: 0.4,
    fontFace: H, fontSize: 18, bold: true, color: RED, isTextBox: true, margin: 0,
  });
  const todo = [
    "No Snapdragon device — AI Hub profiling harness is built and one command away",
    "Reasoning path still uses the grounded extractive backend, not the 3B LLM",
    "Marathi, Bengali, Punjabi have no working checkpoint — the UI says so",
    "Document scanning uses the OS engine, not our own NPU graphs",
  ];
  s.addText(todo.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < todo.length - 1 } })), {
    x: 6.9, y: 1.8, w: 5.7, h: 2.4,
    fontFace: B, fontSize: 13, color: INK, isTextBox: true, margin: 0, paraSpaceAfter: 8,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 4.5, w: 11.9, h: 1.75, fill: { color: TEAL_SOFT }, rectRadius: 0.12,
  });
  s.addText("Every fallback is labelled", {
    x: 1.1, y: 4.7, w: 10.8, h: 0.4,
    fontFace: H, fontSize: 19, bold: true, color: TEAL, isTextBox: true, margin: 0,
  });
  s.addText(
    "The whole product runs on a reviewer's laptop with zero models downloaded. Anything on a " +
    "fallback path is marked degraded in the API and on screen. SETU never passes a stub off as " +
    "a real inference — in a clinical tool, that distinction is the difference between a demo and a lie.",
    {
      x: 1.1, y: 5.1, w: 10.8, h: 1.0,
      fontFace: B, fontSize: 13.5, color: TEAL, isTextBox: true, margin: 0, lineSpacingMultiple: 1.25,
    }
  );
  s.addNotes("Naming your own gaps buys more credibility than claiming perfection.");
}

function closeSlide() {
  const s = pres.addSlide();
  s.background = { color: INK };

  s.addText("Not a translator.\nA check that the patient understood.", {
    x: M, y: 1.5, w: 10.5, h: 1.8,
    fontFace: H, fontSize: 40, bold: true, color: PAPER, isTextBox: true, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addText("Offline, because it has to be.", {
    x: M, y: 3.3, w: 10.5, h: 0.5,
    fontFace: B, fontSize: 20, color: "6EE7B7", isTextBox: true, margin: 0,
  });

  const facts = [
    ["9", "models orchestrated"],
    ["3", "settings, one engine"],
    ["12", "Indian languages"],
    ["99", "tests passing"],
  ];
  let x = M;
  for (const [big, label] of facts) {
    s.addText(big, {
      x: x, y: 4.4, w: 2.9, h: 0.7,
      fontFace: H, fontSize: 40, bold: true, color: PAPER, isTextBox: true, margin: 0,
    });
    s.addText(label, {
      x: x, y: 5.1, w: 2.9, h: 0.35,
      fontFace: B, fontSize: 13, color: "8EA0B8", isTextBox: true, margin: 0,
    });
    x += 3.05;
  }

  s.addText("Full source, architecture notes, benchmarks and evaluation in the repository.", {
    x: M, y: 6.1, w: 11.9, h: 0.4,
    fontFace: B, fontSize: 14, color: "CFD8E3", isTextBox: true, margin: 0,
  });
  s.addNotes("Close on the one-line claim.");
}

titleSlide();
problemSlide();
solutionSlide();
momentSlide();
domainsSlide();
snapdragonSlide();
routerSlide();
proofSlide();
statusSlide();
closeSlide();

pres.writeFile({ fileName: "submission/SETU_pitch.pptx" }).then(() => {
  console.log("wrote submission/SETU_pitch.pptx");
});
