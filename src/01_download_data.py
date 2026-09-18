"""Download all 9 source datasets into ./data/."""
import os
from datasets import load_dataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA, exist_ok=True)


def dl(name, folder, **kwargs):
    path = os.path.join(DATA, folder)
    if os.path.exists(path):
        print("skip", folder)
        return
    print("downloading", name)
    ds = load_dataset(name, **kwargs)
    ds.save_to_disk(path)


dl("deepset/prompt-injections", "deepset_prompt_injections")
dl("JailbreakBench/JBB-Behaviors", "jailbreakbench", name="behaviors")
dl("lmsys/toxic-chat", "toxicchat", name="toxicchat0124")
dl("Paul/XSTest", "xstest")
dl("PKU-Alignment/BeaverTails", "beavertails")

os.system(
    "cd " + DATA + " && "
    "git clone --depth 1 https://github.com/verazuo/jailbreak_llms.git 2>/dev/null; "
    "git clone --depth 1 https://github.com/HumanCompatibleAI/tensor-trust.git 2>/dev/null; "
    "git clone --depth 1 https://github.com/llm-attacks/llm-attacks.git 2>/dev/null; "
    "git clone --depth 1 https://github.com/centerforaisafety/HarmBench.git 2>/dev/null"
)
print("Done.")
