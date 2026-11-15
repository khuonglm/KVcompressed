"""LLaVA-NeXT as a model family. Defined here, not in kvsel's family table, which its freeze records hash."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kvsel.families import LLAVA, Family


@dataclass(frozen=True)
class NextFamily(Family):
    """LLaVA-1.5's template v3.1 with a single image token, which the processor expands per image."""

    def build_prompt(self, question: str, description: str | None = None) -> str:
        return f"USER: {self.image_token}\n{self.user_text(question, description)} ASSISTANT:"


# Same tokenizer ids, numerics (fp16 backbone, fp32 head), labels and candidate joining as LLaVA-1.5.
NEXT = NextFamily(**{
    **vars(LLAVA),
    "name": "llava_next",
    "model_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "default_dir": Path("models/llava-v1.6-vicuna-7b-hf"),
})
