# app/llm.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

class MistralLLM:
    def __init__(self):
        model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

        print("[+] Loading Mistral model... (this may take a few seconds)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",  # uses GPU if available
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )

    def generate(self, prompt: str) -> str:
        result = self.pipeline(prompt)[0]["generated_text"]
        # Strip prompt if it echoes back
        return result[len(prompt):].strip() if result.startswith(prompt) else result.strip()


# 🔁 Singleton instance
llm = MistralLLM()

def call_mistral(prompt: str) -> str:
    return llm.generate(prompt)
