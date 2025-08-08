from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import re, torch
from .embedding import embedder, index_name, pc

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").eval()
import torch
print("🚀 CUDA available?", torch.cuda.is_available())
print("📦 Model loaded to:", model.device)

index = pc.Index(index_name)

def retrieve_docs(query, top_k=3):
    q_embed = embedder.encode([query])[0]
    result = index.query(vector=q_embed.tolist(), top_k=top_k, include_metadata=True)
    valid_chunks = []
    for match in result["matches"]:
        chunk_id = int(match["id"].split("-")[1])
        valid_chunks.append(match["metadata"].get("text", ""))  # Optional
    return result["matches"], valid_chunks


def format_prompt(context, question):
  prompt = f"""
  You are an expert insurance assistant. Your task is to analyze insurance policy text and answer user questions in 2 lines using clause-based reasoning.

  Follow this process strictly:
  Step 1: Identify relevant clause(s) from the context.
  Step 2: Think step-by-step about what the clause says — durations, conditions, exceptions.
  Step 3: Provide the final answer clearly in 2 lines.

  Do NOT REPEAT THE PROMPT MESSAGE OR THE EXAMPLES.

  ONLY RETURN THE FINAL STRUCTURED ANSWER with these 3 fields:

  **Decision:** <short and factual answer>  
  **Justification:** <clear reasoning based on clause>  
  **Clause used:** <exact matching clause or quote>  

  Begin your output with <think> and end with </think> to ensure proper reasoning is triggered.

  EXAMPLE:

  Q: What is the waiting period for pre-existing diseases?  
  <think>
  Step-by-step:
  Step 1: Clause says “The waiting period for pre-existing diseases shall be 36 months…”
  Step 2: That implies coverage begins only after 36 months of continuous coverage.

  **Decision:** Not covered until 36-month waiting period is completed  
  **Justification:** Coverage begins only after 36 months, as stated in clause.  
  **Clause used:** “The waiting period for pre-existing diseases shall be 36 months of continuous coverage…”  
  </think>

  ---

  Now answer this actual question:


  ---
  Context:
  \"\"\"{context}\"\"\"


  REMEMBER NOT TO COPY ANYTHING FROM ABOVE.
  Question: {question}

  <think>
  Step-by-step reasoning:
  """

  return prompt

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids): self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] in self.stop_ids

def generate_answer(context, question):
    prompt = format_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stop_ids = tokenizer("</think>", add_special_tokens=False).input_ids
    stopping = StoppingCriteriaList([StopOnTokens(stop_ids)])

    outputs = model.generate(
        **inputs, max_new_tokens=256,
        temperature=0.6, top_k=20, top_p=0.9,
        repetition_penalty=1.1,
        stopping_criteria=stopping
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def clean_model_output(raw_output):
    return raw_output.split("<think>")[-1].strip()

def extract_clean_answer(raw_output):
    return {
        "Decision": re.search(r"\*\*Decision:\*\*\s*(.+)", raw_output).group(1).strip() if re.search(r"\*\*Decision:\*\*", raw_output) else "N/A",
        "Justification": re.search(r"\*\*Justification:\*\*\s*(.+)", raw_output).group(1).strip() if re.search(r"\*\*Justification:\*\*", raw_output) else "N/A",
        "Clause used": re.search(r"\*\*Clause used:\*\*\s*(.+)", raw_output).group(1).strip() if re.search(r"\*\*Clause used:\*\*", raw_output) else "N/A"
    }