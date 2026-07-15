from typing import List

# num_merges isn't the vocabulary size itself, but it's the number of new tokens added 
# on top of the base character set. In real BPE implementations (like GPT's tokenizer), 
# this is exactly how it works — you pick a target vocabulary size, subtract the base 
# character count, and that difference is your num_merges

# The base character set is just the starting vocabulary — the set of unique individual 
# characters that appear in your corpus before any merges happen. 
# E.g. chars = list(corpus)
# target_vocab_size = 10
# num_merges = target_vocab_size - len(set(corpus))  # 10 - 4 = 6

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = list(corpus)
        res = []

        curr_stream = chars
        for _ in range(num_merges):
            freq = {}
            most_freq_pair = None
            most_freq_cnt = float('-inf')

            for j in range(1, len(curr_stream)):
                curr_pair =  (curr_stream[j-1], curr_stream[j])
                freq[curr_pair] = 1 if curr_pair not in freq else (freq[curr_pair]+1) 
                
                if (freq[curr_pair] >  most_freq_cnt) or ((freq[curr_pair] == most_freq_cnt) and (curr_pair < most_freq_pair)):
                    most_freq_pair, most_freq_cnt =  curr_pair, freq[curr_pair]
            
            # most_freq_cnt = max(freq.values())
            # most_freq_pair = sorted(p for p, c in freq.items() if c == most_freq_cnt)[0]
            
            res.append(list(most_freq_pair))

            temp_stream = []
            j = 0

            while j < len(curr_stream)-1:
                curr_pair =  (curr_stream[j], curr_stream[j+1])
                if curr_pair == most_freq_pair:
                    temp_stream.append("".join(curr_pair))
                    j += 2
                else:
                    temp_stream.append(curr_stream[j])
                    j += 1
            
            while j < len(curr_stream):
                temp_stream.append(curr_stream[j])
                j += 1

            curr_stream = temp_stream

        return res