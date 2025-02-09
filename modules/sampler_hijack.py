import json
import math
import pprint
import random

import torch
import transformers
from transformers import LogitsWarper
from transformers.generation.logits_process import (
    LogitNormalization,
    LogitsProcessor,
    LogitsProcessorList
)

from modules import shared
from modules.logging_colors import logger
from modules.models import get_device

global_scores = None


class TemperatureLogitsWarperCustom(LogitsWarper):
    '''
    A copy of the original Transformers temperature logits warper.
    '''

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."

            raise ValueError(except_msg)

        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class DynamicTemperatureLogitsWarper(LogitsWarper):
    '''
    Dynamic temperature.
    '''

    def __init__(self, dynatemp_low: float, dynatemp_high: float, dynatemp_exponent: float):
        self.dynatemp_low = dynatemp_low
        self.dynatemp_high = dynatemp_high
        self.dynatemp_exponent = dynatemp_exponent

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        min_temp = self.dynatemp_low
        max_temp = self.dynatemp_high
        exponent_val = self.dynatemp_exponent

        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)

        # Calculate entropy of the softmax probabilities
        entropy = -1.0 * torch.where(probs > 0, probs * torch.log(probs), torch.zeros_like(probs)).sum()

        # Guard against future possible division by zero
        entropy = max(entropy, torch.tensor(1e-10))  # Ensures entropy is slightly greater than 0

        # Any logits which are not -Infinity will be considered for calculating max entropy.
        num_valid_tokens = torch.sum(scores > -float('inf')).item()

        # Now, calculate the max entropy by using only the valid tokens' count
        max_entropy = math.log(num_valid_tokens)

        # Guard against future possible division by zero
        max_entropy = max_entropy if max_entropy > 0.0 else 1e-10

        # Normalize the entropy
        normalized_entropy = entropy / max_entropy

        # Map the normalized entropy to the desired temperature range using the power function
        dyn_temp = min_temp + (max_temp - min_temp) * (normalized_entropy.pow(exponent_val))

        # Apply the dynamically calculated temperature scaling
        scores = scores / dyn_temp

        # print("----------------------\nTemperature from generation_config:", self.temperature)
        # print("min_temp:", min_temp)
        # print("max_temp:", max_temp)
        # print("Entropy:", entropy.item())
        # print("Max Possible Entropy considering valid tokens only:", max_entropy)
        # print("Normalized Entropy:", normalized_entropy.item())
        # print("Dynamic Temperature (dyn_temp):", dyn_temp.item())
        # print("----------------------")

        # max_prob_token_id = torch.argmax(scores, dim=-1)  # Get the token ID with the highest probability
        # max_prob_token = shared.tokenizer.convert_ids_to_tokens(int(max_prob_token_id))  # Convert ID to token
        # print("--- T=", float(dyn_temp), "token=", max_prob_token, "min=", min_temp, "max=", max_temp, "exponent=", exponent_val)

        return scores


class QuadraticSamplingLogitsWarper(LogitsWarper):
    '''
    Quadratic sampling with smoothing factor and smoothing curve parameters.
    '''

    def __init__(self, smoothing_factor, smoothing_curve):
        self.smoothing_factor = smoothing_factor
        self.smoothing_curve = smoothing_curve

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # Compute necessary values
        max_logit = scores.max()
        diff = scores - max_logit
        k = (3 - self.smoothing_curve) / 2
        s = (self.smoothing_curve - 1) / 2

        # Apply transformation to non-negative infinity values
        transformed_logits = torch.where(
            scores != float('-inf'),
            -(k * self.smoothing_factor * diff**2) + (s * self.smoothing_factor * diff**3) + max_logit,
            scores
        )

        return transformed_logits


class TailFreeLogitsWarper(LogitsWarper):
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float >= 0 and <= 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold (token with 0 are kept)
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs

        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsWarper(LogitsWarper):
    def __init__(self, top_a: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_a = float(top_a)
        if top_a < 0 or top_a > 1.0:
            raise ValueError(f"`top_a` has to be a float >= 0 and <= 1, but is {top_a}")
        self.top_a = top_a
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Remove tokens with probability less than top_a*(max(probs))^2 (token with 0 are kept)
        probs_max = probs[..., 0, None]
        sorted_indices_to_remove = probs < probs_max * probs_max * self.top_a

        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


# Exclude Top Choices (XTC)
class XTCLogitsWarper(LogitsWarper):
    def __init__(self, threshold: float, probability: float, filter_value: float = -float("Inf")):
        self.threshold = threshold
        self.probability = probability
        self.filter_value = filter_value
        self.special_token_ids = [
            shared.tokenizer.encode("\n")[-1],
        ]

        if shared.tokenizer.eos_token_id is not None:
            self.special_token_ids.append(shared.tokenizer.eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # `random` returns values in the half-open range [0, 1), so setting `probability`
        # to 0 means the sampler never takes action, while setting it to 1 means the sampler
        # always takes action.
        #
        # Note that while XTC is most intuitively described as "if multiple tokens meet
        # the threshold, then with probability...", reversing the two conditions is logically
        # equivalent, and improves performance because processing can immediately be stopped
        # if the random check fails.
        if random.random() >= self.probability:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        sorted_indices_to_remove = torch.full_like(probs, False, dtype=torch.bool)

        # This operation sets exactly those indices to `True` for which the next index has
        # probability above the threshold. Since `probs` is sorted, those are the indices
        # of all tokens that meet the threshold, *except* the least probable one.
        sorted_indices_to_remove[..., :-1] = probs[..., 1:] >= self.threshold

        # Convert sorted_indices_to_remove to the original indices
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        # If newline or EOS tokens would be removed, return the original scores
        if indices_to_remove[:, self.special_token_ids].any():
            return scores

        # Otherwise, remove tokens with the mask
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class DRYLogitsProcessor(LogitsProcessor):
    def __init__(self, multiplier: float, base: float, allowed_length: int, sequence_breakers: set[int], _range: int):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._range > 0:
            input_ids = input_ids[:, -self._range:]

        for input_ids_row, scores_row in zip(input_ids, scores):
            # Use normal Python data types for improved performance
            input_ids = input_ids_row.tolist()

            last_token = input_ids[-1]
            if last_token in self.sequence_breakers:
                continue

            # Exclude the last token as it always matches.
            match_indices = []
            for idx, val in enumerate(input_ids[:-1]):
                if val == last_token:
                    match_indices.append(idx)

            # Stores the maximum matching sequence length
            # for each token immediately following the sequence in the input.
            match_lengths = {}

            for i in match_indices:
                next_token = input_ids[i + 1]

                if next_token in self.sequence_breakers:
                    continue

                # We have already found that `last_token` matches at this index,
                # so the match is at least of length 1.
                match_length = 1

                # Extend the match backwards (at most to 50 to prevent exponent overflow at penalty calculation) (this cap also improves performance on worst case)
                while match_length < 50:
                    j = i - match_length
                    if j < 0:
                        # Start of input reached.
                        break

                    previous_token = input_ids[-(match_length + 1)]
                    if input_ids[j] != previous_token:
                        # Start of match reached.
                        break

                    if previous_token in self.sequence_breakers:
                        # Sequence-breaking token reached.
                        break

                    match_length += 1

                if next_token in match_lengths:
                    match_lengths[next_token] = max(match_length, match_lengths[next_token])
                else:
                    match_lengths[next_token] = match_length

            # Apply penalties.
            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_length:
                    penalty = self.multiplier * self.base ** (match_length - self.allowed_length)
                    scores_row[token] -= penalty

        return scores


class MirostatLogitsWarper(LogitsWarper):
    def __init__(self, mirostat_mode: int, mirostat_tau: float, mirostat_eta: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if mirostat_mode not in [2]:
            raise ValueError(f"`mirostat` has to be a an integer 2, but is {mirostat_mode}")

        self.mirostat_mode = mirostat_mode
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.mu = 2 * self.mirostat_tau
        self.e = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores[0]
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()  # candidates

        # Truncate the words with surprise values greater than mu
        for i, candidate in enumerate(prob_original):
            if candidate > 0 and -math.log2(candidate) > self.mu:
                if (i == 0):
                    sorted_logits = sorted_logits[:1]
                else:
                    sorted_logits = sorted_logits[:i]
                break

        # Normalize the probabilities of the remaining words
        prob_topk = torch.softmax(sorted_logits, dim=0)
        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
        device = get_device()
        if device:
            prob_topk = prob_topk.to(device)
            prev_i = prev_i.to(device)

        observed_surprise = -math.log2(prob_topk[prev_i])
        self.e = observed_surprise - self.mirostat_tau

        # Update mu using the learning rate and error
        self.mu -= self.mirostat_eta * self.e

        sorted_indices_to_remove = torch.ones_like(scores[0], dtype=torch.bool)
        sorted_indices_to_remove[prev_i] = False

        indices_to_remove = sorted_indices_to_remove.unsqueeze(0).scatter(1, sorted_indices.unsqueeze(0), sorted_indices_to_remove.unsqueeze(0))
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class SpyLogitsWarper(LogitsWarper):
    def __init__(self):
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global global_scores
        global_scores = scores
        return scores


class RepetitionPenaltyLogitsProcessorWithRange(LogitsProcessor):
    def __init__(self, penalty: float, _range: int):
        if not (penalty > 0):
            raise ValueError(f"`penalty` has to be strictly positive, but is {penalty}")
        self.penalty = penalty
        self._range = _range

    def apply_repetition_penalty(self, input_ids_row, scores_row):
        unique_ids = torch.unique(input_ids_row)
        score = torch.gather(scores_row, 0, unique_ids)

        # Apply multiplicative repetition penalty
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores_row.scatter_(0, unique_ids, score)
        return scores_row

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, -self._range:]
        for input_ids_row, scores_row in zip(input_ids, scores):
            scores_row = self.apply_repetition_penalty(input_ids_row, scores_row)

        return scores


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, presence_penalty: float, _range: int):
        self.presence_penalty = presence_penalty
        self._range = _range

    def apply_presence_penalty(self, input_ids_row, scores_row):
        unique_ids, counts = torch.unique(input_ids_row, return_counts=True)

        # Apply presence penalty
        raw_presence_penalty = (counts > 0).to(scores_row.dtype)
        presence_penalty = raw_presence_penalty * self.presence_penalty
        scores_row.scatter_add_(0, unique_ids, -presence_penalty)
        return scores_row

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, -self._range:]
        for input_ids_row, scores_row in zip(input_ids, scores):
            scores_row = self.apply_presence_penalty(input_ids_row, scores_row)
        return scores


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, frequency_penalty: float, _range: int):
        self.frequency_penalty = frequency_penalty
        self._range = _range

    def apply_frequency_penalty(self, input_ids_row, scores_row):
        unique_ids, counts = torch.unique(input_ids_row, return_counts=True)

        # Apply frequency penalty
        raw_frequency_penalty = counts.to(scores_row.dtype)
        frequency_penalty = raw_frequency_penalty * self.frequency_penalty
        scores_row.scatter_add_(0, unique_ids, -frequency_penalty)
        return scores_row

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids[:, -self._range:]
        for input_ids_row, scores_row in zip(input_ids, scores):
            scores_row = self.apply_frequency_penalty(input_ids_row, scores_row)
        return scores

# === 新增：Margin Adaptive Logits Warper ===
class MarginAdaptiveLogitsWarper(LogitsWarper):
    """
    Margin Adaptive Logits Warper

    此採樣器根據最高與次高 logits 之間的 margin 動態調整溫度。
    計算公式：
         f(margin) = min_factor + (1 - min_factor) * (min(margin, threshold) / threshold)
         T_effective = base_temperature * f(margin)
         
    當 margin 較小（模型不夠自信）時，降低有效溫度，使分布更集中；當 margin 較大時則保持基礎溫度。
    
    參數：
        base_temperature (float): 初始溫度，通常取自 generation_config.temperature。
        threshold (float): 當 margin 超過此值後不再進行調整。
        min_factor (float): 當 margin 為 0 時的最小縮放因子。
    """
    def __init__(self, base_temperature: float, threshold: float = 1.0, min_factor: float = 0.5):
        self.base_temperature = base_temperature
        self.threshold = threshold
        self.min_factor = min_factor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, _ = torch.sort(scores, descending=True, dim=-1)
        top_logit = sorted_logits[..., 0]
        second_logit = sorted_logits[..., 1] if scores.size(-1) > 1 else top_logit
        margin = top_logit - second_logit
        clipped_margin = torch.clamp(margin, max=self.threshold)
        scaling_factor = self.min_factor + (1.0 - self.min_factor) * (clipped_margin / self.threshold)
        effective_temperature = self.base_temperature * scaling_factor
        effective_temperature = torch.clamp(effective_temperature, min=1e-6)
        return scores / effective_temperature

# === 新增：Chain-of-Thought Refinement 採樣器 ===
class ChainOfThoughtRefinementLogitsWarper(LogitsWarper):
    """
    Chain-of-Thought Refinement Logits Warper

    此採樣器透過多次梯度更新，
    將 softmax 分布的熵調整至目標值，從而延長推理時間，
    使模型更充分「思考」後再選擇下一個 token。

    更新規則:
        p = softmax(L)
        E = - sum(p * log(p + epsilon))
        grad = p * (log(p + epsilon) - E)
        L_new = L + step_size * (E - target_entropy) * grad

    參數：
        target_entropy (float): 目標熵 (例如 0.5)
        max_iter (int): 更新次數
        step_size (float): 更新步長
    """
    def __init__(self, target_entropy: float = 0.5, max_iter: int = 3, step_size: float = 0.05):
        self.target_entropy = target_entropy
        self.max_iter = max_iter
        self.step_size = step_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        epsilon = 1e-10
        L = scores.clone()
        for _ in range(self.max_iter):
            p = torch.softmax(L, dim=-1)
            E = -torch.sum(p * torch.log(p + epsilon), dim=-1, keepdim=True)
            grad = p * (torch.log(p + epsilon) - E)
            update = self.step_size * (E - self.target_entropy) * grad
            L = L + update
        return L


# === 新增：MCTS 採樣器 (內化 MCTS 並存取 input_ids) ===
class MCTSSamplerLogitsWarper(LogitsWarper):
    """
    MCTSSamplerLogitsWarper

    此採樣器融合 MCTS 思想，從 logits 中選取 top-K 候選 token，
    並對每個候選 token 進行固定深度 rollout 模擬。 rollout 時，除了累加候選 token 的原始分數外，
    還會根據 input_ids 中上下文中出現的頻率計算一個懲罰值（context penalty），以降低重複出現 token 的得分。
    
    參數：
        candidate_count (int): 每次從 logits 中選取候選 token 數量。
        rollout_depth (int): 每個候選 token 的 rollout 深度。
        rollout_samples (int): 每個候選 token rollout 的次數。
        exploration_const (float): 探索常數 (此處未使用，可擴展)。
    """
    def __init__(self, candidate_count: int = 5, rollout_depth: int = 2, rollout_samples: int = 1, exploration_const: float = 1.0, context_window: int = 10, penalty_factor: float = 0.1):
        self.candidate_count = candidate_count
        self.rollout_depth = rollout_depth
        self.rollout_samples = rollout_samples
        self.exploration_const = exploration_const
        self.context_window = context_window
        self.penalty_factor = penalty_factor

    def compute_context_penalty(self, candidate: torch.Tensor, input_ids: torch.LongTensor) -> float:
        # 從 input_ids 取最後 context_window 個 token
        context = input_ids[0, -self.context_window:]
        count = (context == candidate).sum().item()
        return count * self.penalty_factor

    def rollout(self, base_scores: torch.FloatTensor) -> float:
        p = torch.softmax(base_scores, dim=-1)
        sampled = torch.multinomial(p, num_samples=1)
        return torch.log(p[sampled] + 1e-10).item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 假設 batch_size = 1，取得前 candidate_count 個候選 token
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        candidate_tokens = sorted_indices[0, :self.candidate_count]  # shape [candidate_count]
        candidate_scores = sorted_logits[0, :self.candidate_count]     # shape [candidate_count]

        candidate_values = []
        for i in range(self.candidate_count):
            total_reward = candidate_scores[i].item()
            # 進行 rollout 模擬
            for _ in range(self.rollout_samples):
                rollout_reward = 0.0
                for _ in range(self.rollout_depth):
                    rollout_reward += self.rollout(scores[0])
                total_reward += rollout_reward
            # 加入上下文懲罰，降低在最近上下文中已出現頻率較高的 token 的得分
            context_penalty = self.compute_context_penalty(candidate_tokens[i], input_ids)
            avg_reward = (total_reward - context_penalty) / (self.rollout_samples + 1)
            candidate_values.append(avg_reward)
        best_index = candidate_values.index(max(candidate_values))
        best_token = candidate_tokens[best_index].unsqueeze(0)
        new_scores = torch.full_like(scores, -float('inf'))
        new_scores[0, best_token] = scores[0, best_token]
        return new_scores

def get_logits_processor_patch(self, **kwargs):
    generation_config = kwargs['generation_config']

    # Parameter sanitization
    if isinstance(generation_config.temperature, int):
        generation_config.temperature = float(generation_config.temperature)  # Must be float

    # Get the original warpers
    warpers = self._get_logits_processor_old(**kwargs)

    for i in range(len(warpers) - 1, -1, -1):
        # Replace temperature with our modified class.
        if warpers[i].__class__.__name__ == 'TemperatureLogitsWarper':
            warpers[i] = TemperatureLogitsWarperCustom(
                generation_config.temperature,
            )

        # Stuff we don't need
        elif warpers[i].__class__.__name__ in ['RepetitionPenaltyLogitsProcessor']:
            del warpers[i]

    # Add custom warpers
    warpers_to_add = LogitsProcessorList()
    min_tokens_to_keep = 2 if generation_config.num_beams > 1 else 1

    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        warpers_to_add.append(
            RepetitionPenaltyLogitsProcessorWithRange(
                penalty=generation_config.repetition_penalty,
                _range=generation_config.repetition_penalty_range
            )
        )

    if generation_config.presence_penalty is not None and generation_config.presence_penalty != 0.0:
        warpers_to_add.append(
            PresencePenaltyLogitsProcessor(
                presence_penalty=generation_config.presence_penalty,
                _range=generation_config.repetition_penalty_range
            )
        )

    if generation_config.frequency_penalty is not None and generation_config.frequency_penalty != 0.0:
        warpers_to_add.append(
            FrequencyPenaltyLogitsProcessor(
                frequency_penalty=generation_config.frequency_penalty,
                _range=generation_config.repetition_penalty_range
            )
        )

    if generation_config.dry_multiplier is not None and generation_config.dry_multiplier > 0.0:
        dry_sequence_breakers = generation_config.dry_sequence_breakers

        # Support both JSON array notation and comma-separated strings.
        if not dry_sequence_breakers.startswith("["):
            dry_sequence_breakers = "[" + dry_sequence_breakers + "]"

        sequence_breaker_strings = json.loads(dry_sequence_breakers)
        # Prefix with 'a' to get the correct encoding of the token at the end of a text.
        sequence_breakers = {
            shared.tokenizer.encode(f'a{s}')[-1] for s in sequence_breaker_strings
        }

        warpers.append(
            DRYLogitsProcessor(
                multiplier=generation_config.dry_multiplier,
                base=generation_config.dry_base,
                allowed_length=generation_config.dry_allowed_length,
                sequence_breakers=sequence_breakers,
                _range=generation_config.repetition_penalty_range,
            )
        )

    if generation_config.tfs is not None and 0.0 <= generation_config.tfs < 1.0:
        warpers_to_add.append(
            TailFreeLogitsWarper(
                tfs=generation_config.tfs,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    if generation_config.top_a is not None and 0.0 < generation_config.top_a <= 1.0:
        warpers_to_add.append(
            TopALogitsWarper(
                top_a=generation_config.top_a,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    if generation_config.xtc_probability is not None and generation_config.xtc_probability > 0:
        warpers_to_add.append(
            XTCLogitsWarper(
                threshold=generation_config.xtc_threshold,
                probability=generation_config.xtc_probability,
            )
        )

    if generation_config.dynamic_temperature:
        warpers_to_add.append(
            DynamicTemperatureLogitsWarper(
                dynatemp_low=generation_config.dynatemp_low,
                dynatemp_high=generation_config.dynatemp_high,
                dynatemp_exponent=generation_config.dynatemp_exponent,
            )
        )

    if generation_config.smoothing_factor > 0:
        warpers_to_add.append(
            QuadraticSamplingLogitsWarper(
                smoothing_factor=generation_config.smoothing_factor,
                smoothing_curve=generation_config.smoothing_curve
            )
        )

    if generation_config.mirostat_mode is not None and generation_config.mirostat_mode == 2:
        warpers_to_add.append(
            MirostatLogitsWarper(
                mirostat_mode=generation_config.mirostat_mode,
                mirostat_eta=generation_config.mirostat_eta,
                mirostat_tau=generation_config.mirostat_tau,
                min_tokens_to_keep=min_tokens_to_keep
            )
        )

    # --- 新增：Margin Adaptive 採樣器 ---
    if generation_config.margin_adaptive:
        warpers_to_add.append(
            MarginAdaptiveLogitsWarper(
                base_temperature=generation_config.temperature,
                threshold=generation_config.margin_threshold,
                min_factor=generation_config.margin_min_factor
            )
        )

    # --- 新增：Chain-of-Thought Refinement 採樣器 ---
    if generation_config.cot_refinement:
        warpers_to_add.append(
            ChainOfThoughtRefinementLogitsWarper(
                target_entropy=generation_config.cot_target_entropy,
                max_iter=generation_config.cot_max_iter,
                step_size=generation_config.cot_step_size
            )
        )

    # --- 新增：MCTS 採樣器 ---
    if generation_config.mcts_enabled:
        warpers_to_add.append(MCTSSamplerLogitsWarper(
            candidate_count=generation_config.mcts_candidate_count,
            rollout_depth=generation_config.mcts_rollout_depth,
            rollout_samples=generation_config.mcts_rollout_samples,
            exploration_const=generation_config.mcts_exploration_const
        ))

    if len(warpers) > 0 and isinstance(warpers[-1], LogitNormalization):
        normalize = warpers.pop(-1)
    else:
        normalize = None

    warpers += warpers_to_add

    # Sort the samplers.
    sampler_priority = generation_config.sampler_priority
    if generation_config.temperature_last:
        for param_name in ['temperature', 'dynamic_temperature', 'quadratic_sampling']:
            if param_name in sampler_priority:
                index = sampler_priority.index(param_name)
                sampler_priority.append(sampler_priority.pop(index))
            else:
                sampler_priority.append(param_name)
    class_name_to_nickname = {
        'DynamicTemperatureLogitsWarper': 'dynamic_temperature',
        'EpsilonLogitsWarper': 'epsilon_cutoff',
        'EtaLogitsWarper': 'eta_cutoff',
        'MinPLogitsWarper': 'min_p',
        'MirostatLogitsWarper': 'mirostat',
        'QuadraticSamplingLogitsWarper': 'quadratic_sampling',
        'TailFreeLogitsWarper': 'tfs',
        'TemperatureLogitsWarperCustom': 'temperature',
        'TopALogitsWarper': 'top_a',
        'TopKLogitsWarper': 'top_k',
        'TopPLogitsWarper': 'top_p',
        'TypicalLogitsWarper': 'typical_p',
        'XTCLogitsWarper': 'xtc',
        'RepetitionPenaltyLogitsProcessorWithRange': 'repetition_penalty',
        'PresencePenaltyLogitsProcessor': 'presence_penalty',
        'FrequencyPenaltyLogitsProcessor': 'frequency_penalty',
        'DRYLogitsProcessor': 'dry',
        'MarginAdaptiveLogitsWarper': 'margin_adaptive',
        'ChainOfThoughtRefinementLogitsWarper': 'cot_refinement',
        'MCTSSamplerLogitsWarper': 'mcts',
        'EncoderRepetitionPenaltyLogitsProcessor': 'encoder_repetition_penalty',
        'NoRepeatNGramLogitsProcessor': 'no_repeat_ngram',
    }
    def custom_sort_key(obj):
        class_name = obj.__class__.__name__
        if class_name not in class_name_to_nickname or class_name_to_nickname[class_name] not in sampler_priority:
            return -1
        return sampler_priority.index(class_name_to_nickname[class_name])
    warpers = sorted(warpers, key=custom_sort_key)
    if shared.args.verbose:
        logger.info("WARPERS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint([x.__class__.__name__ for x in warpers])
        print()
    if normalize is not None:
        warpers.append(normalize)
    warpers.append(SpyLogitsWarper())
    warpers = LogitsProcessorList(warpers)
    return warpers


def generation_config_init_patch(self, **kwargs):
    self.__init___old(**kwargs)
    self.min_p = kwargs.pop("min_p", 0.0)
    self.dynamic_temperature = kwargs.pop("dynamic_temperature", False)
    self.dynatemp_low = kwargs.pop("dynatemp_low", 1)
    self.dynatemp_high = kwargs.pop("dynatemp_high", 1)
    self.dynatemp_exponent = kwargs.pop("dynatemp_exponent", 1)
    self.smoothing_factor = kwargs.pop("smoothing_factor", 0.0)
    self.smoothing_curve = kwargs.pop("smoothing_curve", 1.0)
    self.tfs = kwargs.pop("tfs", 1.0)
    self.top_a = kwargs.pop("top_a", 0.0)
    self.mirostat_mode = kwargs.pop("mirostat_mode", 0)
    self.mirostat_eta = kwargs.pop("mirostat_eta", 0.1)
    self.mirostat_tau = kwargs.pop("mirostat_tau", 5)
    self.repetition_penalty_range = kwargs.pop("repetition_penalty_range", 0)
    self.presence_penalty = kwargs.pop("presence_penalty", 0)
    self.frequency_penalty = kwargs.pop("frequency_penalty", 0)
    self.dry_multiplier = kwargs.pop("dry_multiplier", 0.0)
    self.dry_base = kwargs.pop("dry_base", 1.75)
    self.dry_allowed_length = kwargs.pop("dry_allowed_length", 2)
    self.dry_sequence_breakers = kwargs.pop("dry_sequence_breakers", '"\\n", ":", "\\"", "*"')
    self.xtc_threshold = kwargs.pop("xtc_threshold", 0.1)
    self.xtc_probability = kwargs.pop("xtc_probability", 0)
    self.temperature_last = kwargs.pop("temperature_last", False)
    self.sampler_priority = kwargs.pop("sampler_priority", [
        'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'dry', 'temperature',
        'dynamic_temperature', 'quadratic_sampling', 'top_k', 'top_p', 'typical_p',
        'epsilon_cutoff', 'eta_cutoff', 'tfs', 'top_a', 'min_p', 'mirostat', 'xtc',
        'margin_adaptive', 'cot_refinement', 'mcts', 'encoder_repetition_penalty', 'no_repeat_ngram'
    ])
    # --- 新增 margin adaptive 相關參數 ---
    self.margin_adaptive = kwargs.pop("margin_adaptive", False)
    self.margin_threshold = kwargs.pop("margin_threshold", 1.0)
    self.margin_min_factor = kwargs.pop("margin_min_factor", 0.5)
    # --- 新增 CoT Refinement 相關參數 ---
    self.cot_refinement = kwargs.pop("cot_refinement", False)
    self.cot_target_entropy = kwargs.pop("cot_target_entropy", 0.5)
    self.cot_max_iter = kwargs.pop("cot_max_iter", 3)
    self.cot_step_size = kwargs.pop("cot_step_size", 0.05)
    # --- 新增 MCTS 相關參數 ---
    self.mcts_enabled = kwargs.pop("mcts_enabled", False)
    self.mcts_candidate_count = kwargs.pop("mcts_candidate_count", 5)
    self.mcts_rollout_depth = kwargs.pop("mcts_rollout_depth", 2)
    self.mcts_rollout_samples = kwargs.pop("mcts_rollout_samples", 1)
    self.mcts_exploration_const = kwargs.pop("mcts_exploration_const", 1.0)


def hijack_samplers():
    transformers.GenerationMixin._get_logits_processor_old = transformers.GenerationMixin._get_logits_processor
    transformers.GenerationMixin._get_logits_processor = get_logits_processor_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch