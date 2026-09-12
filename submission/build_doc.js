// Brief Project Description for the Snapdragon AI Lab Build & Present Challenge.
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
} = require("docx");
const fs = require("fs");

const TEAL = "0F7B6C";
const INK = "0D1420";
const MUTED = "5B6672";
const RED = "B3261E";

const H = "Cambria";
const B = "Calibri";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 140, line: opts.line ?? 276 },
    alignment: opts.align,
    children: [
      new TextRun({
        text,
        font: opts.font ?? B,
        size: opts.size ?? 21, // half-points: 21 = 10.5pt
        bold: opts.bold,
        italics: opts.italics,
        color: opts.color ?? INK,
      }),
    ],
  });
}

function rich(runs, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 140, line: 276 },
    children: runs.map(
      (r) =>
        new TextRun({
          text: r.text,
          font: r.font ?? B,
          size: r.size ?? 21,
          bold: r.bold,
          italics: r.italics,
          color: r.color ?? INK,
        })
    ),
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: H, size: 30, bold: true, color: TEAL })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: H, size: 24, bold: true, color: INK })],
  });
}

function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: "setu-bullets", level: 0 },
    spacing: { after: 90, line: 276 },
    children: [
      new TextRun({ text, font: B, size: 21, color: opts.color ?? INK, bold: opts.bold }),
    ],
  });
}

function rule() {
  return new Paragraph({
    spacing: { before: 80, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "DFE4EA" } },
    children: [new TextRun({ text: "" })],
  });
}

const COLS = [3200, 6100];
function kvTable(rows) {
  return new Table({
    columnWidths: COLS,
    rows: rows.map(
      ([k, v], i) =>
        new TableRow({
          children: [
            new TableCell({
              width: { size: COLS[0], type: WidthType.DXA },
              shading: { type: ShadingType.CLEAR, fill: i % 2 ? "FFFFFF" : "F4F6F8" },
              margins: { top: 90, bottom: 90, left: 140, right: 140 },
              children: [p(k, { bold: true, size: 20, after: 0 })],
            }),
            new TableCell({
              width: { size: COLS[1], type: WidthType.DXA },
              shading: { type: ShadingType.CLEAR, fill: i % 2 ? "FFFFFF" : "F4F6F8" },
              margins: { top: 90, bottom: 90, left: 140, right: 140 },
              children: [p(v, { size: 20, after: 0, color: MUTED })],
            }),
          ],
        })
    ),
  });
}

const doc = new Document({
  creator: "SETU",
  title: "SETU — Brief Project Description",
  numbering: {
    config: [
      {
        reference: "setu-bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 420, hanging: 220 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 }, // US Letter
          margin: { top: 1180, bottom: 1180, left: 1180, right: 1180 },
        },
      },
      children: [
        // ---------------------------------------------------------- header
        new Paragraph({
          spacing: { after: 40 },
          children: [
            new TextRun({ text: "SETU", font: H, size: 56, bold: true, color: INK }),
            new TextRun({ text: "   सेतु · the bridge", font: B, size: 20, color: MUTED }),
          ],
        }),
        p("An offline comprehension engine for India's language gap", {
          font: H, size: 26, color: TEAL, after: 60,
        }),
        p("Brief Project Description — Snapdragon® AI Lab Build & Present Challenge", {
          size: 19, color: MUTED, after: 60,
        }),
        rule(),

        // ---------------------------------------------------------- problem
        h1("1. The problem"),
        p(
          "A patient at an Indian district hospital is told, in English, that they have " +
          "hypertension. Take amlodipine 5 mg od. Get a fasting lipid profile. Come back " +
          "immediately if you get chest pain."
        ),
        p("They nod. They say “haan ji.” They go home.", { italics: true, color: TEAL }),
        p(
          "They did not understand “od”. They did not understand “fasting”. And they do not " +
          "know that chest pain means return now rather than at the follow-up."
        ),
        p(
          "This is not a translation problem — a phone can translate. It is a comprehension " +
          "problem, and its defining feature is that nobody in the room ever finds out it " +
          "happened. The WHO puts adherence to long-term therapy in developing countries at " +
          "roughly 50%; “the patient never understood the instruction” is a large and " +
          "unglamorous share of that."
        ),

        // --------------------------------------------------------- solution
        h1("2. What SETU does"),
        p(
          "A laptop sits on the clinician's desk and listens to the whole consultation with " +
          "the network disabled. The patient never touches it and needs no phone — their " +
          "interface is a printed card."
        ),
        bullet("Transcribes each turn and shows it in the patient's own language."),
        bullet("Flags the jargon the patient almost certainly did not parse — “od”, “fasting”, “lipid profile” — with a plain-language gloss the clinician can read aloud."),
        bullet("Extracts the care plan as it is given: medicines, tests, follow-up, red flags, lifestyle."),
        bullet("Runs a teach-back check — the patient repeats the plan in their own words, and SETU verifies it against what was actually said."),
        bullet("Prints a take-home card in the patient's language, ordered by consequence."),
        p(
          "The fourth step is the product. It is delivered while the patient is still in the " +
          "room:",
          { after: 100 }
        ),
        rich(
          [
            { text: "“2 of 5 understood. ", bold: true, color: RED, size: 22 },
            {
              text: "The patient did not repeat back: if you get chest pain or breathlessness, " +
                "come immediately to the emergency.”",
              color: RED, size: 22,
            },
          ],
          { after: 160 }
        ),
        p(
          "Anyone can build a translator. SETU is a comprehension check — closer to a clinical " +
          "safety instrument than to a language tool.",
          { italics: true }
        ),

        // ---------------------------------------------------------- domains
        h1("3. One engine, three settings"),
        p(
          "Strip the vocabulary away and the clinic, the classroom and the government counter " +
          "are the same problem: two people, a knowledge gap, one side issuing instructions, " +
          "and nobody checking they landed. SETU ships one engine and three domain profiles."
        ),
        kvTable([
          ["Clinic — doctor → patient", "Catches od, fasting, lipid profile. Card ordered with red flags first."],
          ["Classroom — teacher → student", "Catches weightage, internal, plagiarism. Card ordered with exam dates first."],
          ["Counter — officer → citizen", "Catches self attested, BPL, acknowledgement. Card ordered with the deadline first."],
        ]),
        p("", { after: 80 }),
        p(
          "Each profile supplies a jargon lexicon, instruction categories ordered by " +
          "consequence, teach-back questions and card headings. Adding a fourth setting — a " +
          "courtroom, a bank, a panchayat — means writing one Domain object, not another " +
          "application. The test suite fails if one domain's vocabulary leaks into another."
        ),

        // ------------------------------------------------------- snapdragon
        h1("4. Why this requires Snapdragon"),
        p("Three reasons, and only the third is a performance argument."),
        rich([{ text: "It is un-cloudable. ", bold: true, color: TEAL }, {
          text: "A consultation is protected health information. “Send the audio to a data " +
            "centre” is not a design choice available to us. On-device is the requirement, " +
            "not the optimisation.",
        }]),
        rich([{ text: "It must work where the link does not. ", bold: true, color: TEAL }, {
          text: "A primary health centre with an intermittent connection is the normal case, " +
            "not the edge case.",
        }]),
        rich([{ text: "The workload is sustained, not bursty. ", bold: true, color: TEAL }, {
          text: "Continuous speech recognition across a six-hour clinic day is precisely what " +
            "an NPU exists for and what a CPU cannot afford on battery.",
        }]),

        h2("Hexa-Router"),
        p(
          "Most on-device demonstrations hard-code “put everything on the NPU”. That is wrong " +
          "for a real workload: the NPU is shared and finite, some operators fall back " +
          "regardless, and the right answer changes the moment the charger comes out."
        ),
        p(
          "Hexa-Router decides placement per model, per request, from four inputs: each " +
          "model's deployment envelope; live power state (AC versus battery, charge level, " +
          "thermal pressure); request priority (interactive, streaming or background); and a " +
          "cost model learned on the specific machine from measured latency and marginal " +
          "energy. It applies hysteresis, because rebuilding a QNN session costs hundreds of " +
          "milliseconds and thrashing on noise would cost more than any placement gain. It " +
          "selects the QNN HTP performance mode per placement — burst for interactive turns, " +
          "sustained_high_performance for all-day streaming, power_saver under battery pressure."
        ),

        // ------------------------------------------------------------ proof
        h1("5. Evidence"),
        p(
          "Most submissions state that they built something. This one measures whether the " +
          "central claim is true, on a labelled corpus of sessions spanning all three settings."
        ),
        kvTable([
          ["0 dangerous misses", "Told the clinician the patient understood when they had not — the error that sends a patient home."],
          ["100% recall on missed items", "Of instructions the learner truly did not restate, the proportion SETU caught."],
          ["86.7% precision", "Of the items flagged as missed, the proportion genuinely missed."],
          ["99 tests passing", "Green on a bare checkout with no model assets downloaded."],
        ]),
        p("", { after: 80 }),
        p(
          "The two errors are not symmetric and are therefore not averaged together. A false " +
          "alarm costs a clinician five seconds of repetition; a dangerous miss costs a " +
          "patient. Thresholds were chosen at the operating point that keeps dangerous misses " +
          "at zero — not the one that maximises F1."
        ),
        rich([{ text: "The evaluation overturned a design decision. ", bold: true }, {
          text: "Replacing keyword matching with sentence embeddings made the check worse " +
            "(precision 81% → 76%): cosine similarity scores an imperative against a " +
            "first-person promise lower than expected, and the covered and missed score " +
            "ranges overlap. The two signals failed on different items, so the shipped scorer " +
            "unions them, reaching 86.7% — better than either alone.",
        }]),

        // ----------------------------------------------------------- status
        h1("6. What runs today, and what does not"),
        h2("Real inference, verified"),
        bullet("Whisper speech recognition, verified end-to-end on real audio."),
        bullet("Live microphone capture → transcript → care plan."),
        bullet("Translation into eight Indian languages (Hindi, Telugu, Tamil, Kannada, Malayalam, Gujarati, Odia, Urdu)."),
        bullet("Sentence embeddings and voice-activity detection."),
        bullet("Hexa-Router with battery draw measured from the Windows discharge counter."),
        h2("Honestly not finished"),
        bullet("No Snapdragon device was available. The AI Hub profiling harness is written and is one command from producing verified on-device figures."),
        bullet("The reasoning path still uses a grounded extractive backend rather than the 3B LLM."),
        bullet("Marathi, Bengali and Punjabi have no working translation checkpoint; they are excluded and the interface labels them “(no translation)”."),
        bullet("Document capture (OCR) is stubbed."),
        p(
          "Every fallback is labelled. The whole product runs on a reviewer's laptop with zero " +
          "models downloaded, and anything on a fallback path is marked degraded in the API " +
          "and on screen. SETU never passes a stub off as a real inference — in a clinical " +
          "tool, that distinction is the difference between a demonstration and a lie.",
          { italics: true }
        ),

        // ------------------------------------------------------------- run
        h1("7. Running it"),
        p("pip install -e .   →   python -m setu.server.app   →   open 127.0.0.1:8756", { font: "Courier New", size: 19 }),
        p(
          "Press “Replay sample visit”, then “Run teach-back”. Verify the machine with " +
          "python -m setu.cli doctor; reproduce the evaluation with python scripts/run_eval.py " +
          "--compare; benchmark with python scripts/run_bench.py.",
          { after: 60 }
        ),
        p(
          "Architecture notes, model provenance and licences, benchmarks and the full " +
          "evaluation are in the repository under docs/.",
          { color: MUTED }
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("submission/SETU_project_description.docx", buf);
  console.log("wrote submission/SETU_project_description.docx");
});
