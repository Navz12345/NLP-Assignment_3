from __future__ import annotations

from sequential_tuning.providers.base import Message, TextGenerationProvider


class LocalHFProvider(TextGenerationProvider):
    def __init__(self, base_model_name: str, adapter_path: str | None = None, hf_token: str | None = None) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )

        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

        self.model = model

    def generate(self, messages: list[Message], temperature: float = 0.0, max_tokens: int = 512) -> str:
        import torch

        chat_messages = [{"role": message.role, "content": message.content} for message in messages]

        inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": False,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

