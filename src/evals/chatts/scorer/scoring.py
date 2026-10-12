# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Vendored from ChatTS evaluation/evaluate_qa.py (NetManAIOps/ChatTS, MIT),
# with the MCQ2 ability removed (datasets A/B never contain it).
# Local changes vs upstream:
#   1. loguru -> stdlib logging (loguru is not a project dependency).
#   2. The RAGAS-backed calculate_ragas_score is stubbed to return (0.0, {}).
#      The 'reason' metric therefore reports 0.0; 'categorical' and 'numerical'
#      are computed exactly as upstream (pure rule-based, no API key needed).
#   3. evaluate_batch_qa(normalize_enumeration=...) can rewrite "(N)" list
#      markers to "N." before scoring; OFF by default (= upstream behaviour).
import json
import os
import logging
import numpy as np
import re
from typing import *
from tqdm import tqdm
import traceback
from multiprocessing import Pool

logger = logging.getLogger("chatts_eval.scoring")

# 'reason'-producing ability types scored by pure string matching (no API key).
# Every other inductive/deductive ability routes its 'reason' score through
# RAGAS, which is stubbed out when no OpenAI key is configured.
RULE_BASED_REASON_ABILITIES = {'causal'}


def calculate_ragas_score(question: str, response: str, label: str):
    """Stub for the upstream RAGAS scorer (rule-based-only evaluation mode).

    The real scorer needs an OpenAI API key + ragas==0.1.9. Returning a zero
    score keeps the inductive/deductive ability types from crashing while
    leaving the rule-based categorical/numerical metrics fully intact.
    """
    return 0.0, {}


def split_sentences(text):
    abbreviations = ['max.', 'eg.', 'Mrs.', 'Dr.', 'Mr.']
    
    for abbr in abbreviations:
        escaped_abbr = re.escape(abbr)
        text = re.sub(escaped_abbr, abbr.replace('.', '<DOT>'), text)

    pattern = r'[.!?。！？,;，；](?!\d)'
    sentences = re.split(pattern, text)
    
    sentences = [s.strip().replace('<DOT>', '.') for s in sentences if s.strip()]
    
    return sentences

def split_period_sentences(text):
    abbreviations = ['max.', 'eg.', 'Mrs.', 'Dr.', 'Mr.']
    
    for abbr in abbreviations:
        escaped_abbr = re.escape(abbr)
        text = re.sub(escaped_abbr, abbr.replace('.', '<DOT>'), text)

    pattern = r'[.。](?!\d)'
    sentences = re.split(pattern, text)
    
    sentences = [s.strip().replace('<DOT>', '.') for s in sentences if s.strip()]
    
    return sentences

def match_metric_name(metric: str, sentence: str) -> bool:
    pattern = r'[^\u4e00-\u9fa5a-zA-Z]'
    sentence = re.sub(pattern, '', sentence).lower()
    metric = re.sub(pattern, '', metric).lower()

    return metric in sentence

def evaluate_trend(answer: str, attribute: dict, cols: List[str]):
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'steady' in attribute['type']:
        if 'steady' in sentences[0]:
            cate_correct = True
    elif 'decrease' in attribute['type']:
        if 'decreas' in sentences[0].lower():
            cate_correct = True
    elif 'increase' in attribute['type']:
        if 'increas' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    # Check start point
    for sentence in sentences:
        float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
        if float_numbers is None or len(float_numbers) == 0:
            continue
        if 'start' in sentence:
            if abs(attribute['start']) < 0.5:
                if abs(float_numbers[0]) < 0.5:
                    num_correct.append(1.0)
                else:
                    num_correct.append(0.0)
            else:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['start']) / abs(attribute['start']))))
            break
    else:
        num_correct.append(0.0)

    # Check amplitude
    if attribute['type'] != 'keep steady':
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'change value' in sentence or 'from left to right' in sentence:
                if abs(attribute['amplitude']) < 0.5:
                    if abs(float_numbers[0]) < 0.5:
                        num_correct.append(1.0)
                    else:
                        num_correct.append(0.0)
                else:
                    num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['amplitude']) / abs(attribute['amplitude']))))
                break
        else:
            num_correct.append(0.0)

    return [cate_correct], num_correct, [], []

def evaluate_season(answer: str, attribute: dict, cols: List[str]):
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'no' in attribute['type']:
        if 'no periodic' in sentences[0].lower():
            cate_correct = True
    else:
        if 'no' not in sentences[0].lower() and 'periodic' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    if attribute['type'] != 'no periodic fluctuation':
        # Check period
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'each period' in sentence:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['period']) / abs(attribute['period']))))
                break
        else:
            num_correct.append(0.0)

        # Check amplitude
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'amplitude' in sentence:
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['amplitude']) / abs(attribute['amplitude']))))
                break
        else:
            num_correct.append(0.0)
    else:
        num_correct = []

    return [cate_correct], num_correct, [], []

def evaluate_noise(answer: str, attribute: dict, cols: List[str]):
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [0.0], [], []

    if 'almost no' in attribute['type']:
        if 'no noise' in sentences[0].lower():
            cate_correct = True
    else:
        if 'noisy' in sentences[0].lower():
            cate_correct = True

    num_correct = []

    # Check period
    if 'noisy' in attribute['type']:
        for sentence in sentences:
            float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
            if float_numbers is None or len(float_numbers) == 0:
                continue
            if 'standard' in sentence.lower() or 'std' in sentence.lower():
                num_correct.append(max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - attribute['std']) / abs(attribute['std']))))
                break
        else:
            num_correct.append(0.0)

    return [cate_correct], num_correct, [], []

def evaluate_local(answer: str, attribute: dict, cols: List[str]):
    cate_correct = []
    num_correct = []

    # Split into facts
    for feat in attribute:
        matched_flag = False
        pos_numerical = 0.0
        amp_numerical = 0.0
        for fact in re.split(r'[;；]', answer):
            sentences = re.split(r'[，。,;；]', fact)
            if type(feat['type']) == str:
                feat['type'] = [feat['type']]
            if any(i in sentences[0].lower() for i in feat['type']):
                # Check period and amplitude
                for sentence in sentences:
                    float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
                    if float_numbers is None or len(float_numbers) == 0:
                        continue
                    if 'position' in sentence.lower() or 'around point' in sentence.lower():
                        if abs(float_numbers[0] - feat['position']) > 64:
                            continue
                        pos_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['position']) / abs(feat['position'])))
                        matched_flag = True
                    if matched_flag and 'amplitude' in sentence.lower():
                        amp_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['amplitude']) / abs(feat['amplitude'])))
                if matched_flag:
                    break
        cate_correct.append(matched_flag)
        num_correct.append(pos_numerical)
        num_correct.append(amp_numerical)

    return cate_correct, num_correct, [], []

def evaluate_local_inductive(answer: str, attribute: dict, cols: List[str]):
    cate_correct = []
    num_correct = []
    reason_correct = []
    reason_details = []

    # Split into facts
    for feat in attribute:
        matched_flag = False
        pos_numerical = 0.0
        amp_numerical = 0.0
        reason_score = 0.0
        cur_detail = {}
        for fact in re.split(r'[;；]', answer):
            sentences = re.split(r'[，。,;；]', fact)
            if type(feat['type']) == str:
                feat['type'] = [feat['type']]
            if any(i in sentences[0].lower() for i in feat['type']):
                # Check period and amplitude
                for sentence in sentences:
                    float_numbers = list(map(float, re.findall(r'-?\d+\.?\d*', sentence)))
                    if float_numbers is None or len(float_numbers) == 0:
                        continue
                    if 'position' in sentence.lower() or 'around point' in sentence.lower():
                        if abs(float_numbers[0] - feat['position']) > 64:
                            continue
                        pos_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['position']) / abs(feat['position'])))
                        matched_flag = True
                    if matched_flag and 'amplitude' in sentence.lower():
                        amp_numerical = max(0.0, min(1.0, 1.0 - abs(float_numbers[0] - feat['amplitude']) / abs(feat['amplitude'])))
                if matched_flag:
                    # Evaluate the inductive
                    reason_score, cur_detail = calculate_ragas_score(
                        question='Please analyze the physical meaning of this local fluctuation in one sentence.',
                        response=split_period_sentences(fact)[-1],
                        label=feat['explain']
                    )
                    cur_detail.update({
                        'label': feat['explain'],
                        'response': split_period_sentences(fact)[-1]
                    })
                    break
        cate_correct.append(matched_flag)
        num_correct.append(pos_numerical)
        num_correct.append(amp_numerical)
        reason_correct.append(reason_score)
        reason_details.append(cur_detail)

    return cate_correct, num_correct, reason_correct, reason_details

def evaluate_shape_correlation_inductive(answer: str, attribute: dict, cols: List[str]):
    cate_correct = False
    sentences = split_sentences(answer)

    if len(sentences) == 0:
        return [False], [], [0.0], [{}]

    if attribute['label']:
        if 'yes' in sentences[0].lower():
            cate_correct = True
    else:
        if 'no' in sentences[0].lower():
            cate_correct = True

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they are correlated/no correlated considering their physical meaning in one sentence.',
                        response=sentences[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]

def evaluate_local_correlation_inductive(answer: str, attribute: dict, cols: List[str]):
    cate_correct = False
    sentences = split_period_sentences(answer)

    if len(sentences) == 0:
        return [False], [], [0.0], [{}]

    if attribute['label']:
        if 'yes' in sentences[0].lower():
            # Check correlation type
            label_cols = set(map(tuple, attribute['pair']))
            answer_cols = set()

            # Split into facts
            for fact in sentences[1].split(';'):
                items = fact.strip().split(',')
                if len(items) == 2:
                    for col in cols:
                        if match_metric_name(col, items[0].strip()):
                            answer_cols.add((col, items[1].strip()))

            if label_cols == answer_cols:
                cate_correct = True
    else:
        if 'no' in sentences[0].lower():
            cate_correct = True

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they are correlated/no correlated considering their physical meaning in one sentence.',
                        response=sentences[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]

def evaluate_shape_cluster_inductive(answer: str, attribute: dict, cols: List[str]):
    cate_correct = 0.0
    num_correct = []

    label_cols = set(attribute['cols'])
    answer_cols = set()

    sentences = split_period_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [], [0.0], [{}]

    # Split into facts
    for fact in split_period_sentences(answer)[0].split(','):
        fact = fact.strip()
        for col in cols:
            if match_metric_name(col, fact):
                answer_cols.add(col)

    # Calculate f1-score for label and answer
    tp = len(label_cols & answer_cols)
    fp = len(answer_cols - label_cols)
    fn = len(label_cols - answer_cols)
    if tp + fp + fn > 0:
        cate_correct = 2 * tp / (2 * tp + fp + fn)

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they have similar overall trend considering their physical meaning in one sentence.',
                        response=split_period_sentences(answer)[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]

def evaluate_local_cluster_inductive(answer: str, attribute: dict, cols: List[str]):
    cate_correct = 0.0
    num_correct = []

    label_cols = set(zip(attribute['cols'], [i[1] for i in attribute['col_idx']]))
    answer_cols = set()

    sentences = split_period_sentences(answer)

    if len(sentences) == 0:
        return [0.0], [], [0.0], [{}]

    # Split into facts
    for fact in split_period_sentences(answer)[0].split(';'):
        items = fact.strip().split(',')
        if len(items) == 2:
            for col in cols:
                if match_metric_name(col, items[0].strip()):
                    answer_cols.add((col, items[1].strip()))

    # Calculate f1-score for label and answer
    tp = len(label_cols & answer_cols)
    fp = len(answer_cols - label_cols)
    fn = len(label_cols - answer_cols)
    if tp + fp + fn > 0:
        cate_correct = 2 * tp / (2 * tp + fp + fn)

    num_correct = []
    reason_correct, reason_detail = calculate_ragas_score(
                        question='Explain why they have similar local fluctuations considering their physical meaning in one sentence.',
                        response=split_period_sentences(answer)[-1],
                        label=attribute['explain']
                    )

    return [cate_correct], num_correct, [reason_correct], [reason_detail]

def evaluate_deductive(answer: str, attribute: str, cols: List[str]):
    labels = split_sentences(attribute)
    sentences = split_sentences(answer)

    cur_reason_correct = 1.0
    if labels[0].lower().strip() in ['yes', 'no']:
        if sentences[0].lower().strip() != labels[0].lower().strip():
            cur_reason_correct = 0.0
        ragas_detail = {"label": labels[0], "response": sentences[0]}
    else:
        ragas_correct, ragas_detail = calculate_ragas_score(
                            question='According to the previous information, please answer Yes or No and explain it in detail.',
                            response=answer,
                            label=attribute
                        )
        cur_reason_correct = ragas_correct
    return [], [], [cur_reason_correct], [ragas_detail]

def evaluate_causal(answer: str, attribute: str, cols: List[str]):
    label = split_sentences(attribute)[0].lower().strip()
    answer_choice = split_sentences(answer)[0].lower().strip()
    if match_metric_name(label, answer_choice):
        reason_correct = 1.0
    else:
        reason_correct = 0.0
    return [], [], [reason_correct], [{'label': label, 'response': answer_choice}]

def ability_type_to_func(ability_type: str):
    return eval("evaluate_" + ability_type.replace('-', '_'))

def evaluate_qa(answer: str, sample: dict):
    answer_list = re.findall(r'(?:^|\n).*?\d+\.\s*(.*?)(?=\n.*?\d+\.|$)', answer, re.MULTILINE | re.DOTALL)
    num_questions = len(sample['attributes'])
    ability_types = sample['ability_types']
    matched_cnt = min(len(answer_list), num_questions)

    # Try match
    if matched_cnt < num_questions and matched_cnt == 1:
        idx_pos = []
        for idx in range(1, num_questions + 1):
            sub_str = f"{idx}. "
            if sub_str in answer:
                idx_pos.append(answer.index(sub_str))
            else:
                break

        if len(idx_pos) == num_questions:
            # Matched
            idx_pos.append(len(answer))
            answer_list = [answer[idx_pos[i] + len(f"{i+1}. "):idx_pos[i + 1]] for i in range(num_questions)]
            matched_cnt = min(len(answer_list), num_questions)
            print("[TRY MATCH]---------------------------")
            print(f"[TRY MATCH] {answer_list}")
            print("[TRY MATCH]---------------------------")
    elif num_questions == 1 and matched_cnt == 1 and len(answer_list[0].strip()) == 0:
        # Empty answer
        answer_list[0] = answer.rstrip()
        print("[TRY MATCH]---------------------------")
        print(f"[TRY MATCH] {answer_list}")
        print("[TRY MATCH]---------------------------")

    result = {}

    for i in range(len(ability_types)):
        ability = ability_types[i]
        evaluate_func = ability_type_to_func(ability)
        cur_answer = answer_list[i] if i < matched_cnt else ""
        cate_correct, num_correct, reason_correct, reason_detail = evaluate_func(cur_answer, sample['attributes'][i], sample['cols'])

        if ability in result:
            # Extent current result to existed
            cate_correct = result[ability][0] + cate_correct
            num_correct = result[ability][1] + num_correct
            reason_correct = result[ability][2] + reason_correct
            reason_detail = result[ability][3] + reason_detail
        result[ability] = (cate_correct, num_correct, reason_correct, reason_detail)  

    return result

def process_sample(args):
    idx, sample, generated_answer = args
    try:
        label = sample['answer']

        # find idx in generated answer
        pos = -1
        for i, item in enumerate(generated_answer):
            if item['idx'] == idx:
                pos = i
                break
        answer = generated_answer[pos]['response']
        evaluation_result = evaluate_qa(answer, sample)

        return {
            'idx': idx,
            'label': label,
            'response': answer,
            'evaluation': evaluation_result
        }
    except Exception as err:
        logger.error(err)
        traceback.print_exc()
        return None

def evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=8, out_dir=None,
                      normalize_enumeration: bool = False):
    if normalize_enumeration:
        # Deliberate LENIENCY beyond the ChatTS authors' protocol (upstream has
        # no such step): rewrite "(N)"-style enumerations to "N." so the
        # authors' "N."-only list regex can align answers that enumerate as
        # "(1) (2) ...". It can only raise a score, never lower it; off by
        # default so results match upstream.
        import re as _re
        for _r in generated_answer:
            if isinstance(_r, dict) and isinstance(_r.get("response"), str):
                _r["response"] = _re.sub(r"(?m)^\((\d+)\)\s*", r"\1. ", _r["response"])
    out_dir = out_dir or f"exp/{EXP}"
    os.makedirs(out_dir, exist_ok=True)
    detailed_result = []
    ability_result = {'categorical': {}, 'numerical': {}, 'reason': {}}
    overall_result = {'categorical': [], 'numerical': [], 'reason': []}

    # Evaluation
    logger.info("Start evaluation...")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, [(idx, dataset[idx], generated_answer) for idx in range(len(dataset))]), total=len(dataset)))

    for result in results:
        if result is None:
            continue

        detailed_result.append(result)
        evaluation_result = result['evaluation']

        # Parse naive result
        for ability, (cate_correct, num_correct, reason_correct, reason_detail) in evaluation_result.items():
            ability_result['categorical'].setdefault(ability, [])
            ability_result['numerical'].setdefault(ability, [])
            ability_result['reason'].setdefault(ability, [])

            ability_result['categorical'][ability].extend(cate_correct)
            ability_result['numerical'][ability].extend(num_correct)
            ability_result['reason'][ability].extend(reason_correct)

            overall_result['categorical'].extend(cate_correct)
            overall_result['numerical'].extend(num_correct)
            overall_result['reason'].extend(reason_correct)

    # Calculate tokens
    total_tokens = 0
    for item in generated_answer:
        total_tokens += item.get('num_tokens', 0)

    logger.info(f"[RESULT] -----------------------------------------------------------------")
    logger.info(f"[RESULT] Experiment: {EXP}")
    logger.info(f"[RESULT] Total: {len(dataset)}, Success Evaluation: {len(detailed_result)}")
    logger.info(f"[RESULT] Detailed Categorical: {[(k, round(float(np.mean(v)), 4)) for (k, v) in ability_result['categorical'].items()]}")
    logger.info(f"[RESULT] Detailed Numerical: {[(k, round(float(np.mean(v)), 4)) for (k, v) in ability_result['numerical'].items()]}")
    logger.info(f"[RESULT] Detailed Reason: {[(k, round(float(np.mean(v)), 4)) for (k, v) in ability_result['reason'].items()]}")
    logger.info(f"[RESULT] Overall Categorical: {round(float(np.mean(overall_result['categorical'])), 4)}; Overall Numerical: {round(float(np.mean(overall_result['numerical'])), 4)}; Overall Reason: {round(float(np.mean(overall_result['reason'])), 4)}")
    logger.info(f"[RESULT] Consumed tokens: {total_tokens}")
    logger.info(f"[RESULT] -----------------------------------------------------------------")

    # Save Result
    def _mean(v):
        return round(float(np.mean(v)), 4) if len(v) else None

    # 'reason' is RAGAS-scored (needs an API key) for every ability type except
    # those in RULE_BASED_REASON_ABILITIES; only the latter aggregate is API-free.
    rb_reason = [x for ab in RULE_BASED_REASON_ABILITIES
                 for x in ability_result['reason'].get(ab, [])]

    json.dump(detailed_result, open(os.path.join(out_dir, "detailed_result.json"), "wt"),
              ensure_ascii=False, indent=4)
    summary = {
        'detail_categorical': dict((k, _mean(v)) for (k, v) in ability_result['categorical'].items()),
        'detail_numerical': dict((k, _mean(v)) for (k, v) in ability_result['numerical'].items()),
        'detail_reason': dict((k, _mean(v)) for (k, v) in ability_result['reason'].items()),
        'overall_categorical': _mean(overall_result['categorical']),
        'overall_numerical': _mean(overall_result['numerical']),
        'overall_reason': _mean(overall_result['reason']),
        'overall_reason_rulebased': _mean(rb_reason),
        'reason_note': ('overall_reason mixes in RAGAS scores stubbed to 0.0 '
                        '(no API key); only overall_reason_rulebased '
                        f'(abilities {sorted(RULE_BASED_REASON_ABILITIES)}) '
                        'is API-free and meaningful.'),
        'consumed_tokens': total_tokens,
    }
    json.dump(summary, open(os.path.join(out_dir, "result.json"), "wt"),
              ensure_ascii=False, indent=4)
    return summary
