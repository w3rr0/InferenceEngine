from typing import cast

from transformers import PreTrainedTokenizerFast  # TODO: should be lazy import


class IncrementalDetokenizer:
    """Decodes generated tokens for one request, emitting only stable deltas."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast, lookback: int = 8):
        self.tokenizer = tokenizer
        self.lookback = lookback
        self.token_ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0

    def _decode(self, token_ids: list[int]) -> str:
        return cast(
            str,
            self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
        )

    def add(self, token_id: int) -> str:
        """Append one token and return the newly stable text (possibly empty)."""
        self.token_ids.append(token_id)

        window = self.token_ids[self.prefix_offset : self.read_offset]
        prefix_text = self._decode(window)
        new_text = self._decode(self.token_ids[self.prefix_offset :])

        if len(new_text) <= len(prefix_text) or new_text.endswith("\ufffd"):
            return ""

        self.read_offset = len(self.token_ids)
        self.prefix_offset = max(0, self.read_offset - self.lookback)

        return new_text[len(prefix_text) :]
