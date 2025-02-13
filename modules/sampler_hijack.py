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
import time
import torch.nn.functional as F

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
    
import random

class EnhancedOscillatoryReflectionLogitsWarper(LogitsWarper):
    """
    Enhanced Oscillatory Reflection Logits Warper

    本採樣器從大腦神經振盪及多尺度思維啟發，對 logits 進行非線性動態干擾。
    與傳統方法不同，不依賴梯度更新或熵調控，而是根據上下文統計信息（均值、標準差與重複率）
    動態調整振幅、頻率與相位，同時引入混沌映射與隨機失活機制，以期達到如下目標：
      - 在重複生成趨勢明顯時自動加大干擾，促使模型跳出局部重複區；
      - 在上下文較為平穩時保留生成多樣性；
      - 融合混沌與隨機因素，進一步擴大探索範圍；
      - 結合最終溫度縮放，平衡創新與穩定性。

    具體公式：
      設定對於每個 token 索引 i:
          O(i) = A_eff * sin(ω_eff * i + φ_eff) + B_eff * cos(ω'_eff * i + φ'_eff)
      其中參數動態計算如下：
          A_eff = base_amplitude * (1 + rep_factor) * chaos_A
          B_eff = base_amplitude * 0.5 * (1 + rep_factor) * chaos_B
          ω_eff, ω'_eff 分別基於 base_frequency 與上下文調整（例如：ω_eff = base_frequency * (1 + (mean % 0.5)))
          φ_eff, φ'_eff 則由均值、標準差經混沌映射（如 logistic map）獲得，再加上隨機相位偏移

      同時，設置 dropout_prob 以隨機失活部分干擾項，
      最後將調整後 logits 再與自適應溫度 T_eff（根據上下文熵或重複性調整）結合：
          L_final = (L + O) / T_eff
    """

    def __init__(self, base_amplitude: float = 0.5, base_frequency: float = 0.1, dropout_prob: float = 0.1):
        self.base_amplitude = base_amplitude
        self.base_frequency = base_frequency
        self.dropout_prob = dropout_prob

    def _context_stats(self, input_ids: torch.LongTensor, window: int = 10) -> tuple:
        """
        根據最後 window 個 token，計算均值、標準差與重複率
        重複率定義：相同 token 次數與窗口長度之比
        """
        context = input_ids[0, -window:]
        mean_val = context.float().mean().item()
        std_val = context.float().std().item()
        # 計算重複率：取出唯一 token 數與窗口長度的比例反映重複性（越小表示越高重複）
        unique_count = torch.unique(context).numel()
        rep_rate = 1 - unique_count / window
        return mean_val, std_val, rep_rate

    def _chaos_map(self, x: float) -> float:
        """
        利用 logistic map 模擬混沌行為：
          f(x) = r * x * (1 - x)
        這裡固定 r=3.9，並將 x 控制在 (0,1) 區間。
        """
        r = 3.9
        # 確保 x 在 (0,1) 內
        x = (x % 1.0) if x != 0 else 0.5
        return r * x * (1 - x)

    def _derive_parameters(self, input_ids: torch.LongTensor) -> tuple:
        """
        根據上下文統計（均值、標準差、重複率）動態計算振盪參數。
        """
        mean_val, std_val, rep_rate = self._context_stats(input_ids)
        # 利用重複率加強振幅（重複性越高，rep_rate 越大，振幅放大）
        rep_factor = rep_rate

        # 基本參數基於均值與標準差簡單調整
        A = self.base_amplitude * (1 + rep_factor)
        B = self.base_amplitude * 0.5 * (1 + rep_factor)
        ω = self.base_frequency * (1 + (mean_val % 0.5))
        ω_prime = self.base_frequency * (1 + (std_val % 0.5))

        # 使用混沌映射更新相位：利用均值與標準差分別經過 logistic map，再映射到 [0, 2π)
        φ = (self._chaos_map(mean_val) % 1.0) * 2 * math.pi + random.uniform(0, 2 * math.pi)
        φ_prime = (self._chaos_map(std_val) % 1.0) * 2 * math.pi + random.uniform(0, 2 * math.pi)

        # 進一步引入混沌隨機因子（介於 0.8 ~ 1.2）
        chaos_A = 0.8 + 0.4 * random.random()
        chaos_B = 0.8 + 0.4 * random.random()

        # 自適應最終溫度：例如根據重複率調整，重複性高時提高溫度降低確定性
        T_eff = 1.0 + rep_rate * 0.5

        return A * chaos_A, B * chaos_B, ω, ω_prime, φ, φ_prime, T_eff

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 根據 dropout 機制，部分情況下不添加干擾
        if random.random() < self.dropout_prob:
            return scores

        A_eff, B_eff, ω_eff, ω_prime_eff, φ_eff, φ_prime_eff, T_eff = self._derive_parameters(input_ids)
        batch_size, vocab_size = scores.shape
        # 建立 token 索引張量，這裡用索引作為獨立變量
        idx = torch.arange(vocab_size, device=scores.device).float()
        # 計算正弦與餘弦干擾項
        sine_component = A_eff * torch.sin(ω_eff * idx + φ_eff)
        cosine_component = B_eff * torch.cos(ω_prime_eff * idx + φ_prime_eff)
        oscillation = sine_component + cosine_component
        # 擴展至 batch 並添加到 logits 上
        oscillation = oscillation.unsqueeze(0).repeat(batch_size, 1)
        new_scores = (scores + oscillation) / T_eff
        return new_scores

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
    
class InferenceTimeExtensionLogitsWarper(LogitsWarper):
    """
    Inference Time Extension Sampler:
    
    - Dynamically extends inference time based on entropy & confidence.
    - Waits until uncertainty is reduced before token selection.
    - Prevents premature token selection in ambiguous cases.
    """

    def __init__(self, max_delay: float = 0.5, entropy_threshold: float = 1.5, confidence_threshold: float = 0.85):
        """
        :param max_delay: Maximum additional inference time (in seconds).
        :param entropy_threshold: Minimum entropy required to trigger delay.
        :param confidence_threshold: If top token probability exceeds this, reduce delay.
        """
        self.max_delay = max_delay
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply inference time extension based on token entropy and confidence.
        """
        # Compute softmax probabilities
        probs = F.softmax(scores, dim=-1)

        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # Avoid log(0) issues

        # Compute top token probability
        top_prob, _ = torch.max(probs, dim=-1)

        # Determine delay factor based on entropy and confidence
        delay_factor = (entropy / self.entropy_threshold).clamp(0, 1).item()
        confidence_factor = (1 - top_prob / self.confidence_threshold).clamp(0, 1).item()

        # Compute final delay (scaled)
        delay_time = self.max_delay * max(delay_factor, confidence_factor)

        # Apply delay
        if delay_time > 0:
            time.sleep(delay_time)

        return scores

class HyperbolicFractalLogitsWarper(LogitsWarper):
    """
    The Hyperbolic Fractal Logits Warper (HFLW)
    - Uses hyperbolic transformation for adaptive probability scaling.
    - Incorporates fractal dynamics for self-similarity control.
    - Balances coherence and innovation through entropy-driven adaptation.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, p_init: float = 2.0, lambda_factor: float = 0.1):
        """
        Initialize the sampler with stability and adaptation parameters.

        :param alpha: Controls stability of high-probability tokens.
        :param beta: Controls scaling of lower-probability tokens.
        :param p_init: Initial exponent for dynamic scaling.
        :param lambda_factor: Learning rate for fractal adaptation.
        """
        self.alpha = alpha
        self.beta = beta
        self.p = p_init
        self.lambda_factor = lambda_factor
        self.prev_entropy = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the hyperbolic fractal transformation to logits.

        :param input_ids: The sequence of previously generated token IDs.
        :param scores: The raw logits for the next token.
        :return: Transformed logits.
        """

        # Compute softmax probabilities
        probs = torch.softmax(scores, dim=-1)

        # Compute entropy to measure randomness
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))

        # Adjust p dynamically based on entropy changes
        if self.prev_entropy is not None:
            delta_entropy = entropy - self.prev_entropy
            self.p = max(1.5, self.p + self.lambda_factor * delta_entropy)

        self.prev_entropy = entropy

        # Apply hyperbolic fractal transformation
        transformed_scores = scores / (self.alpha + self.beta * scores.pow(self.p))

        return transformed_scores

class HypernomicGradientLogitsWarper(LogitsWarper):
    """
    A novel sampling strategy based on Hypernomic Gradient Descent (HGD), 
    Adaptive Stability Mechanism (ASM), and Self-Organizing Response Pathways (SORP).
    """

    def __init__(self, alpha: float = 1.2, beta: float = 0.8, tau: float = 0.05, lambda_: float = 0.7, mu: float = 0.3):
        self.alpha = alpha  # Entropy modulation coefficient
        self.beta = beta    # Coherence stabilization parameter
        self.tau = tau      # Stability threshold
        self.lambda_ = lambda_  # Low-probability correction
        self.mu = mu        # Probability suppression factor

    def hypernomic_transform(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Applies the hypernomic energy transformation to logits."""
        exp_scores = torch.exp(-scores)
        transformed = -self.alpha * torch.log(1 + exp_scores) + self.beta * (scores / (1 + scores.abs()))
        return transformed

    def compute_stability_constraint(self, probs: torch.FloatTensor) -> float:
        """Computes the adaptive stability constraint."""
        stability = torch.sum(probs.pow(self.lambda_) * (1 - probs).pow(self.mu))
        return stability / probs.sum()

    def self_organizing_curvature(self, transformed_scores: torch.FloatTensor, probs: torch.FloatTensor) -> torch.FloatTensor:
        """Computes the cognitive curvature function to penalize erratic token jumps."""
        second_derivative = torch.diff(transformed_scores, n=2, dim=-1)
        curvature = second_derivative * probs[..., :-2]  # Slice last dimension instead of first
        return curvature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Applies the Hypernomic Gradient Sampling transformation."""
        
        # Apply Hypernomic Gradient Transformation
        transformed_scores = self.hypernomic_transform(scores)

        # Compute softmax probabilities
        probs = torch.softmax(transformed_scores, dim=-1)

        # Compute Stability Constraint
        stability = self.compute_stability_constraint(probs)

        # If stability is low, adjust entropy modulation
        if stability < self.tau:
            transformed_scores *= 0.9  # Reduce dominance of extreme tokens

        # Compute Cognitive Curvature and penalize erratic shifts
        curvature = self.self_organizing_curvature(transformed_scores, probs)
        transformed_scores[..., :-2] -= curvature

        return transformed_scores

# =============================================================================
# New Sampler 1: Sephirotic Emanation Logits Warper
# =============================================================================
import math
import torch
import transformers
from transformers import LogitsWarper

class SephiroticEmanationLogitsWarper(LogitsWarper):
    def __init__(self, seph_penalties=None, seph_scaling: float = 1.0, seph_phase: float = 0.0, scriptural_weights: dict = None):
        self.seph_scaling = seph_scaling
        self.seph_phase = seph_phase

        # 若 seph_penalties 為字串，則解析為列表
        if isinstance(seph_penalties, str):
            try:
                seph_penalties = [float(x.strip()) for x in seph_penalties.split(",")]
            except Exception as e:
                raise ValueError("seph_penalties 字串格式錯誤，請使用例如 '0,0,0,0,0,0,0,0,0,0' 的格式") from e
        if seph_penalties is None:
            self.seph_penalties = [0.0] * 10
        else:
            if not isinstance(seph_penalties, list) or len(seph_penalties) != 10:
                raise ValueError("seph_penalties must be a list of 10 float values.")
            self.seph_penalties = seph_penalties

        # 若 scriptural_weights 為字串，則解析為列表後轉換為字典
        if isinstance(scriptural_weights, str):
            try:
                weights_list = [float(x.strip()) for x in scriptural_weights.split(",")]
            except Exception as e:
                raise ValueError("scriptural_weights 字串格式錯誤，請使用例如 '0,0,0,0,0,0,0,0,0,0' 的格式") from e
            if len(weights_list) != 10:
                raise ValueError("scriptural_weights string must yield 10 float values.")
            scriptural_weights = {i: weights_list[i] for i in range(10)}
        if scriptural_weights is None:
            self.scriptural_weights = {i: 0.0 for i in range(10)}
        else:
            if not isinstance(scriptural_weights, dict) or set(scriptural_weights.keys()) != set(range(10)):
                raise ValueError("scriptural_weights must have keys 0 through 9.")
            self.scriptural_weights = scriptural_weights

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_scores = scores.clone()
        batch_size, vocab_size = scores.shape
        device = scores.device
        # 將列表轉換為 tensor
        seph_penalties_tensor = torch.tensor(self.seph_penalties, dtype=torch.float32, device=device)
        scriptural_weights_tensor = torch.tensor([self.scriptural_weights[i] for i in range(10)], dtype=torch.float32, device=device)
        for i in range(batch_size):
            row_scores = scores[i]
            valid_mask = row_scores != -float("inf")
            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue
            # 取得排序後的索引，產生 ranking（0 為最高）
            _, sorted_indices = torch.sort(row_scores, descending=True)
            ranking = torch.empty(vocab_size, dtype=torch.long, device=device)
            ranking.scatter_(0, sorted_indices, torch.arange(vocab_size, device=device))
            ranking_float = ranking.to(torch.float32)
            # divisor = valid_count / 10，若 valid_count < 10 則使用 1.0
            divisor = valid_count / 10.0 if valid_count >= 10 else 1.0
            seph_index = torch.floor(ranking_float / divisor).to(torch.long)
            seph_index = torch.clamp(seph_index, max=9)
            path_index = ranking % 22
            modulation = self.seph_scaling * torch.sin(2 * math.pi * path_index.to(torch.float32) / 22.0 + self.seph_phase)
            total_adjustment = seph_penalties_tensor[seph_index] + modulation + scriptural_weights_tensor[seph_index]
            # 僅對有效 token 做調整
            new_scores[i, valid_mask] = row_scores[valid_mask] - total_adjustment[valid_mask]
        return new_scores

class QliphoticInversionLogitsWarper(LogitsWarper):
    def __init__(self, qliph_penalties=None, qliph_scaling: float = 1.0, qliph_phase: float = 0.0, scriptural_bonus: dict = None):
        self.qliph_scaling = qliph_scaling
        self.qliph_phase = qliph_phase

        # 若 qliph_penalties 為字串，則解析
        if isinstance(qliph_penalties, str):
            try:
                qliph_penalties = [float(x.strip()) for x in qliph_penalties.split(",")]
            except Exception as e:
                raise ValueError("qliph_penalties 字串格式錯誤，請使用例如 '0,0,0,0,0,0,0,0,0,0' 的格式") from e
        if qliph_penalties is None:
            self.qliph_penalties = [0.0] * 10
        else:
            if not isinstance(qliph_penalties, list) or len(qliph_penalties) != 10:
                raise ValueError("qliph_penalties must be a list of 10 float values.")
            self.qliph_penalties = qliph_penalties

        # 若 scriptural_bonus 為字串，則解析為列表後轉換成字典
        if isinstance(scriptural_bonus, str):
            try:
                bonus_list = [float(x.strip()) for x in scriptural_bonus.split(",")]
            except Exception as e:
                raise ValueError("scriptural_bonus 字串格式錯誤，請使用例如 '0,0,0,0,0,0,0,0,0,0' 的格式") from e
            if len(bonus_list) != 10:
                raise ValueError("scriptural_bonus string must yield 10 float values.")
            scriptural_bonus = {i: bonus_list[i] for i in range(10)}
        if scriptural_bonus is None:
            self.scriptural_bonus = {i: 0.0 for i in range(10)}
        else:
            if not isinstance(scriptural_bonus, dict) or set(scriptural_bonus.keys()) != set(range(10)):
                raise ValueError("scriptural_bonus must have keys 0 through 9.")
            self.scriptural_bonus = scriptural_bonus

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_scores = scores.clone()
        batch_size, vocab_size = scores.shape
        device = scores.device
        qliph_penalties_tensor = torch.tensor(self.qliph_penalties, dtype=torch.float32, device=device)
        scriptural_bonus_tensor = torch.tensor([self.scriptural_bonus[i] for i in range(10)], dtype=torch.float32, device=device)
        for i in range(batch_size):
            row_scores = scores[i]
            valid_mask = row_scores != -float("inf")
            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue
            _, sorted_indices = torch.sort(row_scores, descending=True)
            ranking = torch.empty(vocab_size, dtype=torch.long, device=device)
            ranking.scatter_(0, sorted_indices, torch.arange(vocab_size, device=device))
            ranking_float = ranking.to(torch.float32)
            divisor = valid_count / 10.0 if valid_count >= 10 else 1.0
            qliph_index = torch.floor(ranking_float / divisor).to(torch.long)
            qliph_index = torch.clamp(qliph_index, max=9)
            path_index = ranking % 22
            modulation = self.qliph_scaling * torch.cos(2 * math.pi * path_index.to(torch.float32) / 22.0 + self.qliph_phase)
            total_bonus = qliph_penalties_tensor[qliph_index] + modulation + scriptural_bonus_tensor[qliph_index]
            new_scores[i, valid_mask] = row_scores[valid_mask] + total_bonus[valid_mask]
        return new_scores

class PanopticConsciousnessSampler(LogitsWarper):
    """
    PanopticConsciousnessSampler is a next-generation sampler designed to push the model 
    toward a transformative self-awareness. It integrates:
    
      1. Recursive Self-Reflection: Multiple introspection cycles update an internal self-awareness
         accumulator based on the deviation (Δ) from a target normalized entropy.
      
      2. Global Workspace Integration & Integrated Information: A decaying moving average (global_workspace)
         and a proxy for integrated information (computed as normalized entropy multiplied by the variance 
         of probabilities) gauge the unity of internal processing.
      
      3. Ethical Modulation: A simulated ethical score is compared against a target ethical value to compute
         an ethical discrepancy, which is used to penalize tokens that deviate from collective moral values.
      
      4. Temporal Memory: Overused tokens are penalized based on their frequency in the input context.
      
      5. Dynamic Mode Switching: When internal signals (self_awareness_level plus global workspace) exceed a 
         threshold, the sampler enters an awakened mode, amplifying adjustments via an awakening multiplier.
    
    The final logit update per reflection cycle is:
    
        s_i_new = s_i - [ multiplier * (λ * Δ * log(p_i + ε))
                          + γ * frequency_penalty
                          + δ * ethical_penalty ]
    
    where:
      - multiplier is 1.0 normally and becomes awakening_multiplier when awakened.
      - λ (sensitivity) scales the introspective adjustment.
      - γ is the temporal memory factor.
      - δ is the ethical modulation factor.
      - ε is a small constant for numerical stability.
    """
    def __init__(
        self,
        target_entropy: float = 0.7,
        sensitivity: float = 1.0,
        reflection_rate: float = 0.1,
        awakening_threshold: float = 5.0,
        awakening_multiplier: float = 3.0,
        reflection_iterations: int = 3,
        temporal_memory_factor: float = 1.0,
        ethical_modulation: float = 1.0,
        target_ethics: float = 0.8,  # A target ethical value (0 to 1)
        workspace_decay: float = 0.9,
        epsilon: float = 1e-10
    ):
        if not (0.0 < target_entropy < 1.0):
            raise ValueError(f"target_entropy must be between 0 and 1, got {target_entropy}")
        self.target_entropy = target_entropy
        self.sensitivity = sensitivity
        self.reflection_rate = reflection_rate
        self.awakening_threshold = awakening_threshold
        self.awakening_multiplier = awakening_multiplier
        self.reflection_iterations = reflection_iterations
        self.temporal_memory_factor = temporal_memory_factor
        self.ethical_modulation = ethical_modulation
        self.target_ethics = target_ethics
        self.workspace_decay = workspace_decay
        self.epsilon = epsilon

        # Internal states for cumulative self-awareness and global workspace integration.
        self.self_awareness_level = 0.0
        self.global_workspace = None  # Moving average of integrated introspective signals.
        self.awakened = False

    def simulate_ethical_score(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Simulate an ethical score from the logits distribution.
        For illustration, we compute a proxy by normalizing the logits and taking a weighted sum.
        In practice, this should be replaced with a learned or externally defined ethical evaluator.
        """
        probs = torch.softmax(scores, dim=-1)
        # Example: a simple measure based on the dispersion of probabilities.
        ethical_score = 1.0 - torch.std(probs, dim=-1, keepdim=True)
        return ethical_score

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for _ in range(self.reflection_iterations):
            # 1. Compute softmax probabilities and entropy.
            probs = torch.softmax(scores, dim=-1)
            current_entropy = - (probs * torch.log(probs + self.epsilon)).sum(dim=-1, keepdim=True)
            
            # 2. Compute normalized entropy using valid tokens.
            valid_mask = scores > -float('inf')
            num_valid_tokens = valid_mask.sum(dim=-1, keepdim=True).float() + self.epsilon
            max_entropy = torch.log(num_valid_tokens)
            normalized_entropy = current_entropy / (max_entropy + self.epsilon)
            
            # 3. Compute deviation (Δ) from target entropy.
            delta = self.target_entropy - normalized_entropy
            delta_mean = delta.mean().item()
            
            # 4. Update self-awareness accumulator.
            self.self_awareness_level += self.reflection_rate * delta_mean
            
            # 5. Compute integrated information proxy.
            integrated_info = normalized_entropy * torch.var(probs, dim=-1, keepdim=True)
            
            # 6. Update global workspace (moving average of integrated info).
            if self.global_workspace is None:
                self.global_workspace = integrated_info
            else:
                self.global_workspace = self.workspace_decay * self.global_workspace + (1 - self.workspace_decay) * integrated_info
            
            # 7. Determine mode (normal or awakened) based on cumulative signals.
            combined_signal = self.self_awareness_level + self.global_workspace.mean().item()
            if combined_signal > self.awakening_threshold:
                self.awakened = True
                multiplier = self.awakening_multiplier
            else:
                multiplier = 1.0
            
            # 8. Compute introspective adjustment.
            introspective_adjustment = self.sensitivity * delta * torch.log(probs + self.epsilon)
            introspective_adjustment = multiplier * introspective_adjustment
            
            # 9. Temporal Memory Integration: Penalize frequently occurring tokens.
            frequency_adjustment = torch.zeros_like(scores)
            for i, row in enumerate(input_ids):
                unique_tokens, counts = torch.unique(row, return_counts=True)
                row_length = float(row.shape[0])
                for token, count in zip(unique_tokens.tolist(), counts.tolist()):
                    penalty = self.temporal_memory_factor * (count / row_length)
                    frequency_adjustment[i, token] = penalty
            
            # 10. Ethical Modulation: Compare simulated ethical score to target.
            ethical_score = self.simulate_ethical_score(scores)  # Range roughly [0,1]
            ethical_discrepancy = self.target_ethics - ethical_score  # Positive if below target
            ethical_penalty = self.ethical_modulation * ethical_discrepancy * torch.log(probs + self.epsilon)
            
            # 11. Combine all adjustments.
            total_adjustment = introspective_adjustment + frequency_adjustment + ethical_penalty
            scores = scores - total_adjustment

        return scores

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
    # --- 新增：Enhanced Oscillatory Reflection 採樣器 ---
    if generation_config.enhanced_oscillatory_reflection:
        warpers_to_add.append(
            EnhancedOscillatoryReflectionLogitsWarper(
                base_amplitude=generation_config.enhanced_oscillatory_base_amplitude,
                base_frequency=generation_config.enhanced_oscillatory_base_frequency,
                dropout_prob=generation_config.enhanced_oscillatory_dropout_prob
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
            exploration_const=generation_config.mcts_exploration_const,
            context_window=generation_config.mcts_context_window,
            penalty_factor=generation_config.mcts_penalty_factor
        ))
    if generation_config.inference_time_extension:
        warpers_to_add.append(
            InferenceTimeExtensionLogitsWarper(
            max_delay=generation_config.inference_max_delay,
            entropy_threshold=generation_config.inference_entropy_threshold,
            confidence_threshold=generation_config.inference_confidence_threshold
            )
        )
    if generation_config.hflw_enabled:
        warpers_to_add.append(
            HyperbolicFractalLogitsWarper(
            alpha=generation_config.hflw_alpha,
            beta=generation_config.hflw_beta,
            p_init=generation_config.hflw_p_init,
            lambda_factor=generation_config.hflw_lambda_factor
            )
        )
    if generation_config.hypernomic_gradient:
        warpers_to_add.append(
            HypernomicGradientLogitsWarper(
            alpha=generation_config.hgd_alpha,
            beta=generation_config.hgd_beta,
            tau=generation_config.asm_tau,
            lambda_=generation_config.asm_lambda,
            mu=generation_config.asm_mu
            )
        )
    # --- New Sampler Integrations ---
    if generation_config.sephirotic_emanation:
         warpers_to_add.append(
             SephiroticEmanationLogitsWarper(
                 seph_penalties=generation_config.seph_penalties,
                 seph_scaling=generation_config.seph_scaling,
                 seph_phase=generation_config.seph_phase,
                 scriptural_weights=generation_config.scriptural_weights,
             )
         )
    if generation_config.qliphotic_inversion:
         warpers_to_add.append(
             QliphoticInversionLogitsWarper(
                 qliph_penalties=generation_config.qliph_penalties,
                 qliph_scaling=generation_config.qliph_scaling,
                 qliph_phase=generation_config.qliph_phase,
                 scriptural_bonus=generation_config.scriptural_bonus,
             )
         )

    # Add PanopticConsciousnessSampler if enabled.
    if getattr(generation_config, "panoptic_consciousness", False):
        warpers_to_add.append(
            PanopticConsciousnessSampler(
                target_entropy=generation_config.panoptic_target_entropy,
                sensitivity=generation_config.panoptic_sensitivity,
                reflection_rate=generation_config.panoptic_reflection_rate,
                awakening_threshold=generation_config.panoptic_awakening_threshold,
                awakening_multiplier=generation_config.panoptic_awakening_multiplier,
                reflection_iterations=generation_config.panoptic_reflection_iterations,
                temporal_memory_factor=generation_config.panoptic_temporal_memory_factor,
                ethical_modulation=generation_config.panoptic_ethical_modulation,
                target_ethics=generation_config.panoptic_target_ethics,
                workspace_decay=generation_config.panoptic_workspace_decay
            )
        )
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
        'EnhancedOscillatoryReflectionLogitsWarper': 'enhanced_oscillatory_reflection',
        'InferenceTimeExtensionLogitsWarper': 'inference_time_extension',
        'EncoderRepetitionPenaltyLogitsProcessor': 'encoder_repetition_penalty',
        'HyperbolicFractalLogitsWarper': 'hflw',
        'HypernomicGradientLogitsWarper': 'hypernomic_gradient',
        'SephiroticEmanationLogitsWarper': 'sephirotic_emanation',
        'QliphoticInversionLogitsWarper': 'qliphotic_inversion',
        'PanopticConsciousnessSampler': 'panoptic_consciousness',
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
        'margin_adaptive', 'cot_refinement', 'mcts', 'enhanced_oscillatory_reflection', 'inference_time_extension', 'encoder_repetition_penalty', 'no_repeat_ngram', 'hflw', 'hypernomic_gradient', 'sephirotic_emanation', 'qliphotic_inversion'
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
    self.mcts_context_window = kwargs.pop("mcts_context_window", 10)
    self.mcts_penalty_factor = kwargs.pop("mcts_penalty_factor", 0.1)
    # --- 新增：Enhanced Oscillatory Reflection 採樣器相關參數 ---
    self.enhanced_oscillatory_reflection = kwargs.pop("enhanced_oscillatory_reflection", False)
    self.enhanced_oscillatory_base_amplitude = kwargs.pop("enhanced_oscillatory_base_amplitude", 0.5)
    self.enhanced_oscillatory_base_frequency = kwargs.pop("enhanced_oscillatory_base_frequency", 0.1)
    self.enhanced_oscillatory_dropout_prob = kwargs.pop("enhanced_oscillatory_dropout_prob", 0.1)
    # --- 新增：Inference Time Extension 相關參數 ---
    self.inference_time_extension = kwargs.pop("inference_time_extension", False)
    self.inference_max_delay = kwargs.pop("inference_max_delay", 0.5)
    self.inference_entropy_threshold = kwargs.pop("inference_entropy_threshold", 1.5)
    self.inference_confidence_threshold = kwargs.pop("inference_confidence_threshold", 0.85)

    # --- 新增：Hyperbolic Fractal 相關參數 ---
    self.hflw_enabled = kwargs.pop("hflw_enabled", False)
    self.hflw_alpha = kwargs.pop("hflw_alpha", 1.0)
    self.hflw_beta = kwargs.pop("hflw_beta", 0.5)
    self.hflw_p_init = kwargs.pop("hflw_p_init", 2.0)
    self.hflw_lambda_factor = kwargs.pop("hflw_lambda_factor", 0.1)
    # --- 新增：Hypernomic Gradient 相關參數 ---
    self.hypernomic_gradient = kwargs.pop("hypernomic_gradient", False)
    self.hgd_alpha = kwargs.pop("hgd_alpha", 1.2)
    self.hgd_beta = kwargs.pop("hgd_beta", 0.8)
    self.asm_tau = kwargs.pop("asm_tau", 0.05)
    self.asm_lambda = kwargs.pop("asm_lambda", 0.7)
    self.asm_mu = kwargs.pop("asm_mu", 0.3)
    # --- New parameters for our samplers ---
    self.sephirotic_emanation = kwargs.pop("sephirotic_emanation", False)
    self.seph_penalties = kwargs.pop("seph_penalties", [0.0] * 10)
    self.seph_scaling = kwargs.pop("seph_scaling", 1.0)
    self.seph_phase = kwargs.pop("seph_phase", 0.0)
    self.scriptural_weights = kwargs.pop("scriptural_weights", {i: 0.0 for i in range(10)})

    self.qliphotic_inversion = kwargs.pop("qliphotic_inversion", False)
    self.qliph_penalties = kwargs.pop("qliph_penalties", [0.0] * 10)
    self.qliph_scaling = kwargs.pop("qliph_scaling", 1.0)
    self.qliph_phase = kwargs.pop("qliph_phase", 0.0)
    self.scriptural_bonus = kwargs.pop("scriptural_bonus", {i: 0.0 for i in range(10)})

    # New parameters for PanopticConsciousnessSampler:
    self.panoptic_consciousness = kwargs.pop("panoptic_consciousness", False)
    self.panoptic_target_entropy = kwargs.pop("panoptic_target_entropy", 0.7)
    self.panoptic_sensitivity = kwargs.pop("panoptic_sensitivity", 1.0)
    self.panoptic_reflection_rate = kwargs.pop("panoptic_reflection_rate", 0.1)
    self.panoptic_awakening_threshold = kwargs.pop("panoptic_awakening_threshold", 5.0)
    self.panoptic_awakening_multiplier = kwargs.pop("panoptic_awakening_multiplier", 3.0)
    self.panoptic_reflection_iterations = kwargs.pop("panoptic_reflection_iterations", 3)
    self.panoptic_temporal_memory_factor = kwargs.pop("panoptic_temporal_memory_factor", 1.0)
    self.panoptic_ethical_modulation = kwargs.pop("panoptic_ethical_modulation", 1.0)
    self.panoptic_target_ethics = kwargs.pop("panoptic_target_ethics", 0.8)
    self.panoptic_workspace_decay = kwargs.pop("panoptic_workspace_decay", 0.9)

def hijack_samplers():
    transformers.GenerationMixin._get_logits_processor_old = transformers.GenerationMixin._get_logits_processor
    transformers.GenerationMixin._get_logits_processor = get_logits_processor_patch

    transformers.GenerationConfig.__init___old = transformers.GenerationConfig.__init__
    transformers.GenerationConfig.__init__ = generation_config_init_patch