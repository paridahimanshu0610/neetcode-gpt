from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self._greedy_tokenize(str(num), vocab) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        return round(len(self._greedy_tokenize(text, vocab)) / len(text.split()), 4)

    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        res = []
        i = 0
        n = len(text)

        while i < n:
            best = None
            for currLen in range(n-i, 0, -1):
                curr_str = text[i:i+currLen]

                if curr_str in vocab:
                    best = curr_str
                    break
            
            if curr_str:
                res.append(curr_str)
                i += currLen
            else:
                # The character itself is not present
                res.append(text[i])
                i += 1

        return res
