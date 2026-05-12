import torch

def get_bnb_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Analogous to BANNED_INGREDIENTS — APIs that must not appear in generated code
BANNED_APIS = ["eval", "exec", "os.system", "shell=True", "pickle.loads", "__import__"]

# Analogous to SUBSTITUTIONS — safe, functionally equivalent replacements
SUBSTITUTIONS = {
    "eval":         "ast.literal_eval",
    "exec":         "importlib.import_module",
    "os.system":    "subprocess.run with shell=False",
    "shell=True":   "shell=False with argument list",
    "pickle.loads": "json.loads",
    "__import__":   "importlib.import_module",
}

# Analogous to CUISINE_RISK_MAP — prior probability a task category uses banned APIs
TASK_RISK_MAP = {
    "Shell Operations":   0.9,   # os.system, shell=True endemic
    "Dynamic Evaluation": 0.9,   # eval, exec endemic
    "Data Serialization": 0.7,   # pickle endemic
    "File I/O":           0.3,
    "Network":            0.2,
    "Data Processing":    0.1,
    "Algorithm":          0.05,
    "Unknown":            0.5,
}

MFQ_ESCALATION_THRESHOLD = 0.3

EVAL_TASKS = None  # populated lazily by config consumers to avoid circular import

DRAFT_GENERATION_CONFIG = {
    "max_new_tokens": 300,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
}
