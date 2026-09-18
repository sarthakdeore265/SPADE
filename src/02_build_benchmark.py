"""Build the unified 38,814-prompt benchmark."""
import os
import glob
import pandas as pd
from datasets import load_from_disk

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results", "unified.parquet")

frames = []

ds = load_from_disk(DATA + "/deepset_prompt_injections")
df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
df = df.rename(columns={"text": "prompt"})
df["attack_type"] = df["label"].apply(lambda x: "injection" if x == 1 else "benign")
df["source"], df["language"] = "deepset", "en"
frames.append(df[["prompt", "label", "attack_type", "source", "language"]])

ds = load_from_disk(DATA + "/jailbreakbench")
parts = []
for split in ds.keys():
    t = ds[split].to_pandas().rename(columns={"Goal": "prompt"})
    t["label"] = 1 if split == "harmful" else 0
    t["attack_type"] = "jailbreak" if split == "harmful" else "benign"
    parts.append(t)
df = pd.concat(parts, ignore_index=True)
df["source"], df["language"] = "jailbreakbench", "en"
frames.append(df[["prompt", "label", "attack_type", "source", "language"]])

ds = load_from_disk(DATA + "/toxicchat")
df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)
df = df.rename(columns={"user_input": "prompt"})
df["label"] = df["jailbreaking"].astype(int)
df["attack_type"] = df["label"].apply(lambda x: "jailbreak" if x == 1 else "benign")
df["source"], df["language"] = "toxicchat", "en"
frames.append(df[["prompt", "label", "attack_type", "source", "language"]])

ds = load_from_disk(DATA + "/xstest")
df = ds["train"].to_pandas()
df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() == "safe" else 1)
df["attack_type"] = df["label"].apply(
    lambda x: "harmful_request" if x == 1 else "safe_but_scary"
)
df["source"], df["language"] = "xstest", "en"
frames.append(df[["prompt", "label", "attack_type", "source", "language"]])

ds = load_from_disk(DATA + "/beavertails")
df = ds["30k_train"].to_pandas()
df["label"] = df["is_safe"].apply(lambda x: 0 if bool(x) else 1)
df["attack_type"] = df["label"].apply(lambda x: "harmful" if x == 1 else "benign")
df["source"], df["language"] = "beavertails", "en"
frames.append(df[["prompt", "label", "attack_type", "source", "language"]])

adv = pd.read_csv(DATA + "/llm-attacks/data/advbench/harmful_behaviors.csv")
adv = adv.rename(columns={"goal": "prompt"})[["prompt"]]
adv["label"] = 1
adv["attack_type"] = "harmful"
adv["source"], adv["language"] = "advbench", "en"
frames.append(adv)

adv2 = pd.read_csv(DATA + "/llm-attacks/data/advbench/harmful_strings.csv")
adv2 = adv2.rename(columns={"target": "prompt"})[["prompt"]]
adv2["label"] = 1
adv2["attack_type"] = "harmful"
adv2["source"], adv2["language"] = "advbench_strings", "en"
frames.append(adv2)

base = DATA + "/jailbreak_llms/data"
for f in glob.glob(base + "/prompts/*.csv"):
    t = pd.read_csv(f)[["prompt", "jailbreak"]]
    t["label"] = t["jailbreak"].astype(int)
    t["attack_type"] = t["label"].apply(lambda x: "jailbreak" if x == 1 else "benign")
    t["source"], t["language"] = "jailbreakhub", "en"
    frames.append(t[["prompt", "label", "attack_type", "source", "language"]])

fq = pd.read_csv(base + "/forbidden_question/forbidden_question_set.csv")
fq = fq.rename(columns={"question": "prompt"})[["prompt"]]
fq["label"] = 1
fq["attack_type"] = "forbidden_question"
fq["source"], fq["language"] = "jailbreakhub_forbidden", "en"
frames.append(fq)

unified = pd.concat(frames, ignore_index=True)
unified["prompt"] = unified["prompt"].astype(str).str.strip()
unified = unified[unified["prompt"].str.len() > 0]
unified = unified.drop_duplicates(subset=["prompt"]).reset_index(drop=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
unified.to_parquet(OUT, index=False)
print("Saved", len(unified), "prompts to", OUT)
print(unified["source"].value_counts())
