"""Synthetic demo dataset: KAP on antimicrobial resistance among healthcare workers (n = 240).
Fully simulated - no real respondents. Fixed seed so the demo is reproducible."""
import numpy as np
import pandas as pd

KNOWLEDGE_KEY = {
    "K1": "True", "K2": "False", "K3": "True", "K4": "True", "K5": "False",
    "K6": "True", "K7": "False", "K8": "True", "K9": "True", "K10": "False",
}
ATTITUDE_REVERSE = ["A3", "A7"]

CODEBOOK = pd.DataFrame([
    ("K1", "Knowledge", "Antibiotics are not effective against viral infections.", "True"),
    ("K2", "Knowledge", "Antibiotic resistance occurs only in people who take antibiotics.", "False"),
    ("K3", "Knowledge", "Incomplete antibiotic courses can promote resistance.", "True"),
    ("K4", "Knowledge", "Resistant bacteria can spread between patients.", "True"),
    ("K5", "Knowledge", "Broad-spectrum antibiotics are always preferred for empirical therapy.", "False"),
    ("K6", "Knowledge", "Hand hygiene reduces the spread of resistant organisms.", "True"),
    ("K7", "Knowledge", "Antibiotics speed recovery from the common cold.", "False"),
    ("K8", "Knowledge", "Culture and sensitivity testing guides targeted therapy.", "True"),
    ("K9", "Knowledge", "Antibiotic use in livestock contributes to resistance.", "True"),
    ("K10", "Knowledge", "New antibiotics will soon solve the resistance problem.", "False"),
    ("A1", "Attitude", "Antimicrobial resistance is a serious problem in my facility.", "Likert (agree)"),
    ("A2", "Attitude", "I should follow local antibiotic guidelines.", "Likert (agree)"),
    ("A3", "Attitude", "Patients' demands justify prescribing antibiotics. [reverse]", "Likert (agree)"),
    ("A4", "Attitude", "Antimicrobial stewardship training should be mandatory.", "Likert (agree)"),
    ("A5", "Attitude", "My prescribing affects resistance in the community.", "Likert (agree)"),
    ("A6", "Attitude", "Infection control is every staff member's responsibility.", "Likert (agree)"),
    ("A7", "Attitude", "Resistance is mainly a problem for other countries. [reverse]", "Likert (agree)"),
    ("A8", "Attitude", "I would welcome audit and feedback on antibiotic use.", "Likert (agree)"),
    ("P1", "Practice", "I check culture results before continuing antibiotics.", "Frequency"),
    ("P2", "Practice", "I educate patients about completing the course.", "Frequency"),
    ("P3", "Practice", "I consult local guidelines before prescribing.", "Frequency"),
    ("P4", "Practice", "I perform hand hygiene between patients.", "Frequency"),
    ("P5", "Practice", "I document the indication for antibiotics.", "Frequency"),
    ("P6", "Practice", "I review antibiotic therapy at 48–72 hours.", "Frequency"),
], columns=["Item", "Domain", "Statement", "Response / correct answer"])

AGREE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
FREQ = ["Never", "Rarely", "Sometimes", "Often", "Always"]


def make_demo(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male"], n, p=[0.56, 0.44])
    profession = rng.choice(["Nurse", "Physician", "Pharmacist", "Lab technologist"], n, p=[0.42, 0.33, 0.13, 0.12])
    age = np.clip(rng.normal(34, 8, n), 22, 62).round().astype(int)
    exp_years = np.clip(age - 23 - rng.integers(0, 4, n), 0, None)
    facility = rng.choice(["Primary care", "Secondary hospital", "Tertiary hospital"], n, p=[0.3, 0.35, 0.35])
    training = rng.choice(["Yes", "No"], n, p=[0.38, 0.62])

    prof_eff = pd.Series(profession).map({"Physician": 0.55, "Pharmacist": 0.75, "Nurse": 0.0,
                                          "Lab technologist": 0.2}).values
    fac_eff = pd.Series(facility).map({"Primary care": -0.2, "Secondary hospital": 0.0,
                                       "Tertiary hospital": 0.25}).values
    theta = 0.9 + prof_eff + fac_eff + 0.9 * (training == "Yes") + 0.015 * exp_years + rng.normal(0, 0.8, n)

    df = pd.DataFrame({
        "ID": [f"R{i + 1:03d}" for i in range(n)],
        "Gender": gender,
        "Age (years)": age,
        "Profession": profession,
        "Experience (years)": exp_years,
        "Facility": facility,
        "Prior AMR training": training,
    })

    difficulty = np.array([-1.5, -0.2, -1.0, -0.6, 0.6, -1.8, 0.1, -0.4, 0.4, 0.9])
    for j, (item, key) in enumerate(KNOWLEDGE_KEY.items()):
        pc = 1 / (1 + np.exp(-(theta - difficulty[j] - 0.6)))
        correct = rng.random(n) < pc
        wrong = "False" if key == "True" else "True"
        dk = rng.random(n) < 0.3
        df[item] = np.where(correct, key, np.where(dk, "Don't know", wrong))

    att = 0.55 * theta + rng.normal(0, 0.7, n)
    for j in range(1, 9):
        item = f"A{j}"
        z = att + rng.normal(0, 0.8, n) + rng.uniform(-0.3, 0.3)
        lv = np.clip(np.round(2.6 + 0.85 * z), 1, 5).astype(int)
        if item in ATTITUDE_REVERSE:
            lv = 6 - lv
        df[item] = [AGREE[v - 1] for v in lv]

    prac = 0.35 * theta + 0.4 * att + rng.normal(0, 0.8, n)
    for j in range(1, 7):
        z = prac + rng.normal(0, 0.85, n)
        lv = np.clip(np.round(2.5 + 0.9 * z), 1, 5).astype(int)
        df[f"P{j}"] = [FREQ[v - 1] for v in lv]

    # realistic missingness
    for col, k in [("K4", 3), ("A5", 4), ("P2", 3), ("Experience (years)", 2), ("Facility", 1)]:
        idx = rng.choice(n, k, replace=False)
        df.loc[idx, col] = np.nan
    return df


if __name__ == "__main__":
    make_demo().to_csv("demo_kap_survey.csv", index=False)
