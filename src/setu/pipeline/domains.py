"""Domain profiles - the same comprehension engine, three settings.

Clinic, classroom and government counter look like different products, but they are one
problem wearing three uniforms:

    two people, a language or knowledge gap, one side issuing instructions,
    the other side expected to act on them - and nobody checking that the
    instructions actually landed.

So the engine is shared and only the vocabulary changes. A Domain supplies four things:

  jargon      terms this setting's expert uses that the other person will not parse
  kinds       the categories instructions fall into, ordered by consequence
  questions   what to ask when checking comprehension
  headings    how the take-away card is laid out

Adding a fourth setting (a courtroom, a bank, a panchayat) means writing one Domain, not
another agent. That is the whole argument for the abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    #: Who is speaking. (expert, learner) - drives the UI labels and the prompts.
    expert: str
    learner: str
    #: Instruction categories, MOST CONSEQUENTIAL FIRST. The card is ordered by this.
    kinds: tuple[str, ...]
    headings: dict[str, str]
    #: regex -> kind, applied to each sentence the expert says
    patterns: tuple[tuple[str, str], ...]
    jargon: dict[str, str]
    questions: dict[str, str]
    card_title: str
    #: The categories where a missed item is serious enough to interrupt the session.
    critical: tuple[str, ...] = ()
    blurb: str = ""

    def heading(self, kind: str) -> str:
        return self.headings.get(kind, kind.title())

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "expert": self.expert,
            "learner": self.learner,
            "kinds": list(self.kinds),
            "headings": self.headings,
            "critical": list(self.critical),
            "jargon_terms": len(self.jargon),
            "blurb": self.blurb,
        }


# ------------------------------------------------------------------------- clinic

CLINIC = Domain(
    key="clinic",
    title="Clinic — doctor and patient",
    expert="doctor",
    learner="patient",
    kinds=("redflag", "medication", "followup", "test", "lifestyle"),
    headings={
        "redflag": "Come back IMMEDIATELY if",
        "medication": "Your medicines",
        "followup": "Come back on",
        "test": "Tests to get done",
        "lifestyle": "At home",
    },
    critical=("medication", "redflag"),
    card_title="Your visit summary",
    blurb="A doctor trained in English explaining a diagnosis to a patient who is not.",
    patterns=(
        (r"\b(if .*(worse|bleeding|chest pain|breathless|faint|vomit|fever above)|emergency|immediately)\b", "redflag"),
        (r"\b(tablet|tab|capsule|cap|syrup|mg|ml|dose|twice|thrice|once a day|bd|tds|od|sos)\b", "medication"),
        (r"\b(come back|follow.?up|review|next week|after \d+ (day|week|month)|revisit)\b", "followup"),
        (r"\b(test|scan|x-?ray|ecg|usg|ultrasound|blood work|cbc|hba1c|lipid|biopsy|sample)\b", "test"),
        (r"\b(avoid|stop|reduce|walk|exercise|diet|salt|sugar|water|rest|smoking|alcohol)\b", "lifestyle"),
    ),
    questions={
        "medication": "Which medicines will you take, and how many times a day?",
        "test": "Which tests will you get done, and do you need to fast?",
        "followup": "When will you come back to see the doctor?",
        "redflag": "What warning signs mean you must come back immediately?",
        "lifestyle": "What changes will you make at home?",
    },
    jargon={
        "hypertension": "high blood pressure",
        "hypotension": "low blood pressure",
        "diabetes mellitus": "high blood sugar",
        "hyperglycemia": "blood sugar that is too high",
        "hypoglycemia": "blood sugar that is too low",
        "anemia": "low haemoglobin, which makes you weak and tired",
        "hb": "haemoglobin, the iron level in your blood",
        "lipid profile": "a blood test for fat and cholesterol",
        "fasting": "with nothing to eat or drink for 8 to 10 hours before",
        "post prandial": "measured about two hours after eating",
        "hba1c": "a blood test showing your average sugar over three months",
        "antibiotic": "medicine that kills the infection",
        "analgesic": "pain relief medicine",
        "antipyretic": "medicine to bring the fever down",
        "prophylaxis": "medicine taken to stop a problem before it starts",
        "bd": "twice a day",
        "tds": "three times a day",
        "od": "once a day",
        "sos": "only if needed",
        "stat": "right now, immediately",
        "npo": "nothing to eat or drink",
        "chronic": "long lasting, needs ongoing care",
        "acute": "sudden and serious",
        "benign": "not cancer, not dangerous",
        "malignant": "cancer",
        "biopsy": "taking a small piece of tissue to test it",
        "ecg": "a heart tracing test",
        "usg": "an ultrasound scan",
        "cbc": "a complete blood count test",
        "renal": "related to the kidneys",
        "hepatic": "related to the liver",
        "cardiac": "related to the heart",
        "edema": "swelling from fluid",
        "dyspnea": "difficulty breathing",
        "syncope": "fainting",
        "titrate": "slowly adjust the dose",
        "adherence": "taking the medicine exactly as told",
        "follow up": "come back for a check",
        "referral": "being sent to another doctor or hospital",
    },
)


# ---------------------------------------------------------------------- classroom

CLASSROOM = Domain(
    key="classroom",
    title="Classroom — teacher and student",
    expert="teacher",
    learner="student",
    kinds=("exam", "assignment", "concept", "resource"),
    headings={
        "exam": "Exam and test dates",
        "assignment": "What you must submit",
        "concept": "What today's class covered",
        "resource": "What to read or bring",
    },
    critical=("exam", "assignment"),
    card_title="What today's class covered",
    blurb="A lecture delivered in English to students who think in their mother tongue. "
    "India teaches engineering and medicine in a language most students learn late.",
    patterns=(
        (r"\b(exam|test|quiz|viva|internal|midterm|semester|marks|weightage|syllabus)\b", "exam"),
        (r"\b(submit|assignment|homework|due|deadline|hand in|lab record|project report)\b", "assignment"),
        (r"\b(remember|important|note that|the key|this means|derive|formula|theorem|define)\b", "concept"),
        (r"\b(read|refer|textbook|chapter|page|bring|reference|library|notes)\b", "resource"),
    ),
    questions={
        "exam": "When is the test, and what will it cover?",
        "assignment": "What do you have to submit, and by when?",
        "concept": "Explain today's main idea in your own words.",
        "resource": "What will you read or bring next time?",
    },
    jargon={
        "weightage": "how many marks it is worth",
        "viva": "a spoken exam where the teacher asks you questions",
        "internal": "a test conducted by the college that counts towards your final marks",
        "prerequisite": "something you must learn first",
        "derive": "work out the formula step by step yourself",
        "theorem": "a rule that has been proved",
        "hypothesis": "an idea you test to see if it is true",
        "empirical": "based on what is actually measured, not theory",
        "iterate": "repeat the steps until it is right",
        "syllabus": "the list of topics the exam can ask about",
        "rubric": "the rules the teacher uses to give marks",
        "plagiarism": "copying someone else's work and calling it yours",
        "cite": "say where you got the information from",
        "abstract": "a short summary at the start",
        "peer review": "other experts checking the work before it is published",
        "asymptotic": "how it behaves when the numbers get very large",
        "deprecated": "old, should not be used any more",
    },
)


# ------------------------------------------------------------------------ counter

COUNTER = Domain(
    key="counter",
    title="Government or bank counter — officer and citizen",
    expert="officer",
    learner="citizen",
    kinds=("deadline", "document", "fee", "eligibility", "nextstep"),
    headings={
        "deadline": "Last date - do NOT miss this",
        "document": "Papers you must bring",
        "fee": "What you must pay",
        "eligibility": "Whether you qualify",
        "nextstep": "What to do next",
    },
    critical=("deadline", "document"),
    card_title="What you need to do",
    blurb="A counter clerk explaining a scheme, a loan, or a rejection to a citizen who "
    "cannot read the form they are being asked to sign.",
    patterns=(
        (r"\b(last date|before \d|deadline|expires|valid till|within \d+ (day|week|month)|due date)\b", "deadline"),
        (r"\b(aadhaar|pan|ration card|photo|passbook|certificate|proof|xerox|photocopy|affidavit|attach|self.?attest)\b", "document"),
        (r"\b(fee|charge|rupees|rs\.?|pay|amount|deposit|instal?ment|penalty|fine)\b", "fee"),
        (r"\b(eligible|qualify|income limit|below poverty|category|reserved|criteria|only if)\b", "eligibility"),
        (r"\b(come back|counter|window|apply|submit|sign|register|portal|office|next)\b", "nextstep"),
    ),
    questions={
        "deadline": "By when must you do this?",
        "document": "Which papers will you bring?",
        "fee": "How much will you pay, and where?",
        "eligibility": "Do you qualify, and why?",
        "nextstep": "What is the next thing you will do?",
    },
    jargon={
        "self attested": "you sign the photocopy yourself to say it is genuine",
        "affidavit": "a written statement you sign in front of a notary",
        "notarised": "stamped by a notary to make it official",
        "gazetted officer": "a senior government officer who can verify your documents",
        "domicile": "proof that you live in this state",
        "ews": "economically weaker section",
        "bpl": "below poverty line",
        "kyc": "proving who you are with ID documents",
        "nominee": "the person who receives it if something happens to you",
        "moratorium": "a period where you do not have to repay",
        "collateral": "property you promise to the bank if you cannot repay",
        "disbursement": "when the money is actually paid to you",
        "sanctioned": "approved",
        "rejected on scrutiny": "turned down when the papers were checked",
        "acknowledgement": "the receipt proving you submitted it",
        "grievance": "an official complaint",
        "subsidy": "money the government pays so it costs you less",
        "installment": "paying part of the amount at a time",
        "penalty": "extra money you pay for being late",
    },
)


DOMAINS: dict[str, Domain] = {d.key: d for d in (CLINIC, CLASSROOM, COUNTER)}
DEFAULT_DOMAIN = "clinic"


def get_domain(key: str | None) -> Domain:
    return DOMAINS.get(key or DEFAULT_DOMAIN, CLINIC)
