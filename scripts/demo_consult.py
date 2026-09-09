from setu.pipeline import ConsultAgent, Engine, TakeHomeCard

agent = ConsultAgent(Engine())
agent.start("demo", patient_language="hi")

script = [
    ("doctor",  "Your BP is high, you have hypertension. We start amlodipine 5 mg tablet, one at night, od."),
    ("patient", "Doctor sahib, kitne din tak leni hai?"),
    ("doctor",  "Take it daily, long term. This is chronic, it needs ongoing care."),
    ("doctor",  "Get a lipid profile and HbA1c done, fasting, before the next visit."),
    ("doctor",  "Come back for follow up after 2 weeks."),
    ("doctor",  "If you get chest pain or breathlessness, come immediately to the emergency."),
    ("doctor",  "Reduce salt in your food and walk for thirty minutes daily."),
]
for speaker, line in script:
    out = agent.add_turn("demo", speaker, line)
    if out["new_glosses"]:
        for term, gloss in out["new_glosses"].items():
            print(f"  [jargon] {term!r} -> {gloss}")

print("\n" + "=" * 60)
print("TEACH-BACK QUESTIONS")
for q in agent.teachback_questions("demo"):
    print(f"  ? {q}")

# The patient repeats back only part of it, which is the realistic case.
result = agent.check_teachback("demo", "Main raat ko tablet lunga. Aur do hafte baad aaunga.")
print(f"\n  covered {result['covered']}/{result['total']}")
print("  MUST BE REPEATED BY THE DOCTOR:")
for text in result["needs_repeat"]:
    print(f"    ! {text}")

print("\n" + "=" * 60)
print(TakeHomeCard(agent).to_text("demo"))
