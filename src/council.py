"""
Council of LLMs for DMRS Defense-Level Classification.

Architecture: Multi-Phase Deliberative Council
===============================================

Phase 1 — Initial Assessment (3 agents, parallel):
  1. Clinical Analyst       psychodynamic reasoning (stressor → function → level)
  2. Mechanism Specialist   systematic mechanism screening across all 9 levels
  3. Pattern Analyst        few-shot analogical reasoning with retrieved examples

  → Unanimous high-confidence → early exit (3 LLM calls)

Phase 2 — Differential Diagnosis (conditional, parallel):
  Collects candidate labels from Phase 1 (primary + alternative labels).
  For each candidate:
    - Retrieves class-specific training examples (most similar within that class)
    - Runs a Class Advocate agent that evaluates fit: STRONG / MODERATE / WEAK

Phase 3 — Resolution:
  a) If exactly one candidate has STRONG fit → pick it (0 calls).
  b) If ambiguous → Pairwise head-to-head comparison of top-2 candidates
     with class-specific examples for EACH side (1 call).
  c) Fall back to weighted vote if all else fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types as genai_types

from .config import (
    DEFENSE_DESCRIPTIONS,
    DEFENSE_LEVELS,
    DEFENSE_MECHANISMS,
    DMRS_Q_ITEMS,
    LABEL_SUMMARY,
    CouncilConfig,
    ModelConfig,
)
from .retriever import ExampleRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_dialogue(sample: dict) -> str:
    """Render dialogue turns into a readable string."""
    lines = []
    for turn in sample["dialogue"]:
        lines.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


def format_few_shot_examples(examples: list[dict]) -> str:
    """Render retrieved examples as a numbered list for the prompt."""
    parts = []
    for i, ex in enumerate(examples, 1):
        conv = format_dialogue(ex)
        parts.append(
            f"--- Example {i} (label: {ex['label']} = {DEFENSE_LEVELS[ex['label']]}) ---\n"
            f"Dialogue:\n{conv}\n"
            f"Target utterance: {ex['current_text']}\n"
        )
    return "\n".join(parts)


def build_level_reference() -> str:
    """Detailed reference card for all 9 levels with DMRS-Q behavioral items."""
    lines = []
    for k in sorted(DEFENSE_LEVELS):
        mechs = ", ".join(DEFENSE_MECHANISMS[k]) if DEFENSE_MECHANISMS[k] else "N/A"
        block = (
            f"Level {k} – {DEFENSE_LEVELS[k]}\n"
            f"  Mechanisms: {mechs}\n"
            f"  Description: {DEFENSE_DESCRIPTIONS[k]}\n"
        )
        # Add DMRS-Q behavioral indicators
        q_items = DMRS_Q_ITEMS.get(k, {})
        for mech_name, items in q_items.items():
            block += f"  {mech_name} indicators:\n"
            for item in items:
                block += f"    - {item}\n"
        lines.append(block)
    return "\n".join(lines)


LEVEL_REFERENCE = build_level_reference()

# ---------------------------------------------------------------------------
# Handbook-derived classification rules (from DMRS Coding Handbook FAQ)
# ---------------------------------------------------------------------------

_HANDBOOK_RULES = """\

CRITICAL CLASSIFICATION RULES (from the DMRS coding handbook):

1. EMOTION ≠ DEFENSE: The direct expression of an emotion (e.g., "I am very sad", \
"I feel anxious") is NOT a defense mechanism. A defense is only present when \
emotional expression is clearly distorted, avoided, or transformed. Simply \
narrating past feelings in a safe therapeutic environment is self-disclosure, \
not a defense.

2. SELF-DISCLOSURE ≠ SELF-ASSERTION (L7): Telling a therapist "I was very sad \
when my girlfriend broke up with me" is NOT self-assertion. Self-assertion is \
expressing feelings *at the moment of conflict to achieve a goal* — e.g., \
"Your unilateral decision makes me feel hurt and I need to express how I feel \
right now." Key test: is the speaker ACTIVELY navigating a current conflict, \
or passively reporting a past experience?

3. MULTIPLE DEFENSES → CHOOSE LOWER LEVEL: If both a lower-level defense (e.g., \
L3 Rationalization) and a higher-level defense (e.g., L7 Self-Observation) are \
plausible, the lower (less mature) level is typically chosen as the primary label.

4. NEGATIVE TALK ≠ AUTOMATIC DEFENSE: Not all negative talk about others is a \
defense. Check: Is the purpose to elevate self / shift blame / avoid guilt \
(= defense)? Or to state a fact / solve a problem / communicate directly \
(= not a defense or mature defense)? Is the negative evaluation exaggerated \
or a misattribution (= defense) or factually based (= not a defense)?

5. "THANK YOU" RULE: A simple "thank you" on its own is L0 (No Defense). However, \
a sincere and detailed expression of gratitude that reflects on the help received \
and builds connection can be L7 (Affiliation).

6. SHORT-TEXT CAUTION: Splitting, Reaction Formation, Repression, Undoing, \
Dissociation, and Autistic Fantasy are very difficult to identify in short \
utterances. Require strong, specific evidence before assigning these mechanisms.

7. L0 CHECKLIST — these are Level 0 ONLY when they serve a purely social/logistical \
function with NO emotional processing or coping:
   - Greetings and social pleasantries ("Hello", "Bye")
   - Simple expressions of thanks ("Thank you")
   - Simple responses: "yes," "no," "okay" (with no emotional elaboration)
   - Asking factual/logistical questions ("When is our next appointment?")
   - Social small talk on neutral topics (weather, traffic)
   IMPORTANT: If the speaker is reporting experiences within an emotional \
   conversation where they are processing feelings, engaging with a stressor, \
   or responding to therapeutic prompts — it is likely NOT L0 even if the \
   utterance itself seems simple. Context matters.
"""

_L7_GATE = """\

L7 CHECK — Before assigning Level 7 (High-Adaptive), verify:
1. The speaker is engaging with a psychological stressor or conflict (not just \
   making small talk or performing social functions).
2. The utterance serves an ACTIVE coping function — the speaker is seeking help, \
   reflecting on their feelings, asserting needs, using humor to diffuse tension, \
   planning ahead, or channeling emotions constructively.
3. You can name the SPECIFIC L7 mechanism (Affiliation, Altruism, Anticipation, \
   Humor, Self-Assertion, Self-Observation, Sublimation, or Suppression).
Note: sharing emotional experiences in a therapeutic context with genuine \
engagement IS often L7 (e.g., Affiliation, Self-Observation). The key \
distinction is whether the speaker is PROCESSING the experience (L7) vs \
just answering a question with no psychological work (L0).
"""

# Commonly confused pairs — injected into Phase 1 prompts to reduce errors
_CONFUSION_WATCHLIST = """\

CONFUSION WATCHLIST — carefully distinguish these commonly confused pairs:

L0 vs L7: Level 0 is ONLY for purely social/logistical utterances with NO \
psychological function (greetings, simple thanks, logistics). If the speaker \
is engaging emotionally with a stressor — even through narration — consider \
L7 or another defense level, not L0.

L6 vs L7 (CRITICAL — most common confusion):
  Obsessional (L6) = speaker discusses emotional content but DRAINS THE AFFECT:
    - Generalizes personal pain ("many people are suffering", "times are weird")
    - Describes emotional experience analytically without expressing proportional feeling
    - REPORTS facts about a distressing situation without emotional processing: \
      stating what happened (job loss, illness, hardship) in a matter-of-fact way \
      is NOT coping — it is Isolation of Affect (L6) or Intellectualization (L6)
    - Uses intellectual distance or abstraction to keep feelings at arm's length
  Adaptive (L7) = speaker RETAINS appropriate emotion while processing:
    - Expresses feelings alongside analysis ("I feel scared but I'm trying to...")
    - Actively seeks connection or solutions with emotional engagement
    - Shows evidence of WORKING THROUGH the emotion, not just cataloguing facts
  Key test: Is the speaker FEELING while they talk, or just REPORTING? \
  Reporting painful facts without proportional emotion → L6. \
  Engaging emotionally with the situation → L7.

L5 (Neurotic) — often missed entirely. The hallmark is a DISCONNECT between \
  affect and content, or between the expected emotional response and what is expressed:
    - Displacement: Speaker redirects from their own distress to ask about \
      others or focus on tangential topics. Signals: "How about you?", "Are you \
      affected by...?", sudden topic change when emotional content arises.
    - Repression: The speaker HAS emotion but CANNOT identify its source or \
      content. Key phrases: "I'm just so sad" (but can't say why), "nothing in \
      particular", "just a bit out of it", vague sadness without identifiable cause. \
      The affect is PRESENT but the IDEAS are BLOCKED.
    - Dissociation: Feeling detached, "spaced out", or confused in response to \
      emotional content — a temporary disruption of awareness.
    - Reaction Formation: Expressing the OPPOSITE of expected feelings — being \
      overly cheerful/helpful when context suggests distress.
  KEY SIGNAL: If a speaker in a clearly distressing context says something \
  minimizing, vague, or deflecting — they are likely L5, not L0 or L7.
  L5 vs L0: L5 occurs IN emotional context (the person IS distressed); L0 has \
  no emotional context at all.
  L5 vs L7: L5 AVOIDS the emotional content (displaces, represses); L7 ENGAGES with it.

L4 vs L7: Minor image-distorting (L4) inflates or deflates worth of self/others \
  as a coping strategy:
    - Self-devaluation: exaggerated negative self-assessment beyond what facts support \
      ("I'm useless", "I'm such a failure")
    - Idealization of others: attributing exaggerated positive qualities ("She is a \
      fighter", "You are so amazing"), as if those qualities will solve problems
    - Omnipotence: asserting unrealistic power or control over the situation
  L7 maintains REALISTIC self-appraisal. L4 DISTORTS it (up or down) to manage distress.
  Key test: Is the self/other evaluation proportionate to reality? If exaggerated → L4.

L2 (Major Image-Distorting) — look for ABSOLUTE, all-or-nothing statements:
    - Splitting of self: total self-rejection with no nuance ("I am a complete failure")
    - Splitting of situation: sweeping negativity with zero acknowledgment of anything positive
    - Abrupt positive→negative or negative→positive shifts about the same person/thing
  L2 vs L4: L2 is ABSOLUTE ("everything/nothing/always/never"); L4 is EXAGGERATED \
  but retains SOME reality testing.
  L2 vs L3: L2 distorts the IMAGE of self/others; L3 distorts FACTS/explanations.

L3 (Disavowal) — the speaker avoids RESPONSIBILITY or REALITY:
    - Denial: refuses to acknowledge something obvious
    - Rationalization: makes excuses, points to external factors to avoid own role
    - Projection: attributes own unacknowledged feelings to others
  L3 vs L7: L3 DEFLECTS from reality; L7 ENGAGES with reality constructively.

L1 vs L3: Action (L1) is impulsive BEHAVIOUR to discharge tension. Disavowal (L3) \
is cognitive avoidance (denial, projection, rationalization). L1 acts; L3 thinks.

L8 is very rare (~2%). Only use when the utterance is genuinely ambiguous AND too \
brief to determine function. Most utterances have enough context to classify.
"""


# ---------------------------------------------------------------------------
# Phase 1: Initial Assessment Agent Prompts
# ---------------------------------------------------------------------------

CLINICAL_ANALYST_SYSTEM = f"""\
You are a senior clinical psychologist specialising in psychodynamic assessment.
Your task is to identify the defense level of a target utterance within a \
supportive conversation, using the Defense Mechanism Rating Scales (DMRS).

Reasoning process:
1. Read the dialogue and identify the salient psychological stressor or context.
2. Determine the *function* of the target utterance for the speaker \
   (what does it achieve psychologically?).
3. Identify the specific DMRS mechanism at work (if any).
4. Map that mechanism to its defense level.

Key principles:
- Function over form: ask what the utterance *does*, not just what it says.
- Always identify a specific mechanism before assigning a level.
- Level 0 is for socially functional utterances with no psychological function \
  (greetings, thanks, backchannels, logistics).
- Provide your best label AND your second-best alternative.
{_HANDBOOK_RULES}{_L7_GATE}{_CONFUSION_WATCHLIST}
End with exactly these 4 lines:
mechanism: <specific DMRS mechanism name, or "None" for Level 0/8>
confidence: <high|medium|low>
label: <0-8>
alternative: <0-8>
"""

MECHANISM_SPECIALIST_SYSTEM = f"""\
You are a DMRS coding specialist who has memorised all mechanism descriptive \
items across all defense levels.

Your task: systematically screen all defense levels and identify which specific \
mechanism best explains the target utterance.

Reasoning steps:
1. Screen ALL 9 levels — do not skip any. For each level, briefly consider \
   whether any of its mechanisms could explain the target utterance.
2. BEFORE settling on L7, explicitly check:
   a. Could this be L6? Is the speaker reporting/analyzing WITHOUT proportional \
      emotion? (Isolation of Affect, Intellectualization)
   b. Could this be L5? Is the speaker vague about their distress, deflecting \
      to other topics, or showing affect disconnected from content? (Repression, \
      Displacement)
   c. Could this be L4? Is the speaker exaggerating qualities of self or others? \
      (Devaluation, Idealization, Omnipotence)
3. List the top 3 candidate mechanisms with specific textual evidence.
4. Rate each candidate's fit (Strong / Moderate / Weak).
5. Select the best-fit mechanism and output its level.

Critical distinctions:
- Rationalization (L3: making excuses, justifying) vs Intellectualization \
  (L6: draining affect from analytical discussion) vs Self-Observation \
  (L7: genuine reflective insight with appropriate affect).
- Acting Out (L1: impulsive, uncontrolled) vs Self-Assertion (L7: constructive \
  expression of needs/boundaries).
- Denial (L3: refusing to acknowledge) vs Suppression (L7: consciously setting \
  aside distress to cope).
- Isolation of Affect (L6: aware but emotionally flat) vs Anticipation \
  (L7: planning ahead with emotional awareness).
- Devaluation (L4: dismissing worth of self/others) vs Projection (L3: \
  attributing own feelings to others).
{_HANDBOOK_RULES}{_L7_GATE}{_CONFUSION_WATCHLIST}
End with exactly these 4 lines:
mechanism: <specific DMRS mechanism name, or "None" for Level 0/8>
confidence: <high|medium|low>
label: <0-8>
alternative: <0-8>
"""

PATTERN_ANALYST_SYSTEM = f"""\
You are an expert in analogical reasoning for psychological assessment.
You will be given labelled examples followed by a new dialogue to classify.

Your task:
1. FIRST, independently analyze the NEW target utterance:
   a. What is the psychological stressor or conflict (if any)?
   b. What is the defensive FUNCTION of the utterance (what does it do \
      for the speaker psychologically)?
   c. Is it truly a defense, or just self-disclosure / social exchange?
2. THEN, for each provided example, note its defensive function and why \
   it was assigned its label.
3. FINALLY, compare: which example's defensive function most closely \
   parallels the new utterance's function?

CRITICAL WARNINGS:
- Do NOT match by topic similarity. Two utterances about the same topic \
  (e.g., both about job loss) can serve COMPLETELY different functions.
- Narrating a past emotion ("I was sad") is NOT the same as actively coping \
  with a current stressor (L7). It may be L0 (self-disclosure).
- If none of the examples closely match the new utterance's function, \
  reason independently from the DMRS definitions rather than forcing a match.
{_HANDBOOK_RULES}{_L7_GATE}
Provide both your best label and an alternative.

End with exactly these 4 lines:
mechanism: <specific DMRS mechanism name, or "None" for Level 0/8>
confidence: <high|medium|low>
label: <0-8>
alternative: <0-8>
"""


# ---------------------------------------------------------------------------
# Phase 2: Class Advocate Prompt Template
# ---------------------------------------------------------------------------

_CLASS_ADVOCATE_TEMPLATE = """\
You are a specialist evaluator for Level {level} — {level_name}.

Description: {description}
{mechanisms_line}

Your job: evaluate whether the target utterance could be classified as \
Level {level}. You will see examples of REAL Level {level} utterances for \
comparison.

Process:
1. Study the provided Level {level} examples — note their common patterns.
2. Analyze the target utterance for evidence of Level {level} characteristics.
3. Rate the fit HONESTLY:
   - STRONG: Clear, specific evidence of Level {level} mechanisms in the \
     utterance. The utterance functions similarly to the provided examples.
   - MODERATE: Partial or suggestive evidence; plausible but not definitive.
   - WEAK: Little or no evidence; unlikely to be Level {level}. The utterance \
     functions fundamentally differently from the examples.
4. Cite specific words/phrases from the utterance as evidence.

Be rigorous — if the fit is weak, say so clearly. Accurate assessment matters \
more than advocacy.

End with exactly these 2 lines:
evidence_summary: <one sentence of key evidence for or against>
fit: <STRONG|MODERATE|WEAK>
"""


def _build_advocate_system(label: int) -> str:
    """Build a class-specific advocate system prompt."""
    mechs = DEFENSE_MECHANISMS.get(label, [])
    mechs_line = f"Key mechanisms: {', '.join(mechs)}" if mechs else ""
    return _CLASS_ADVOCATE_TEMPLATE.format(
        level=label,
        level_name=DEFENSE_LEVELS[label],
        description=DEFENSE_DESCRIPTIONS[label],
        mechanisms_line=mechs_line,
    )


def _build_advocate_prompt(
    sample: dict, label: int, class_examples: list[dict],
) -> str:
    """Build the user prompt for a class advocate."""
    dialogue = format_dialogue(sample)
    if class_examples:
        ex_text = format_few_shot_examples(class_examples)
    else:
        ex_text = "(No training examples available for this class.)"
    return (
        f"=== Examples of Level {label} ({DEFENSE_LEVELS[label]}) ===\n"
        f"{ex_text}\n\n"
        f"=== Dialogue to evaluate ===\n"
        f"{dialogue}\n\n"
        f"Target utterance (by {sample['dialogue'][-1]['speaker']}):\n"
        f"{sample['current_text']}\n\n"
        f"Does this target utterance fit Level {label} ({DEFENSE_LEVELS[label]})?\n"
        f"Analyze carefully and be honest about the fit."
    )


# ---------------------------------------------------------------------------
# Phase 3: Pairwise Resolution Prompt (used when advocates are ambiguous)
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM = f"""\
You are a specialist in differential diagnosis of DMRS defense levels.

You will compare a target utterance against exactly two candidate defense \
levels, each illustrated with REAL training examples from that class.

Your task:
1. Study the examples of Candidate A — understand their common defensive function.
2. Study the examples of Candidate B — understand their common defensive function.
3. Compare the target utterance's function to BOTH sets of examples.
4. Determine which candidate the target utterance is more functionally similar \
   to (in terms of psychological/defensive function, NOT surface topic).
5. Commit to one candidate.

Focus on FUNCTION: what does the utterance DO for the speaker psychologically? \
Two utterances about different topics can serve the same defensive function.

If one candidate is L7 and the other is a lower level, apply extra scrutiny to \
L7: verify the speaker is ACTIVELY coping with a stressor, not merely narrating. \
If both candidates fit equally, prefer the lower (less mature) level per DMRS rules.
{_HANDBOOK_RULES}
End with exactly:
reasoning: <one sentence explaining your choice>
label: <the chosen level number, 0-8>
"""


def _build_pairwise_prompt(
    sample: dict,
    label_a: int,
    label_b: int,
    examples_a: list[dict],
    examples_b: list[dict],
) -> str:
    """Build the user prompt for pairwise comparison of two candidate levels."""
    dialogue = format_dialogue(sample)

    ex_a_text = format_few_shot_examples(examples_a) if examples_a else "(no examples)"
    ex_b_text = format_few_shot_examples(examples_b) if examples_b else "(no examples)"

    mechs_a = DEFENSE_MECHANISMS.get(label_a, [])
    mechs_b = DEFENSE_MECHANISMS.get(label_b, [])
    mechs_a_str = ", ".join(mechs_a) if mechs_a else "N/A"
    mechs_b_str = ", ".join(mechs_b) if mechs_b else "N/A"

    return (
        f"Dialogue:\n{dialogue}\n\n"
        f"Target utterance (by {sample['dialogue'][-1]['speaker']}):\n"
        f"{sample['current_text']}\n\n"
        f"=== Candidate A: Level {label_a} — {DEFENSE_LEVELS[label_a]} ===\n"
        f"Description: {DEFENSE_DESCRIPTIONS[label_a]}\n"
        f"Mechanisms: {mechs_a_str}\n"
        f"Training examples of Level {label_a}:\n{ex_a_text}\n\n"
        f"=== Candidate B: Level {label_b} — {DEFENSE_LEVELS[label_b]} ===\n"
        f"Description: {DEFENSE_DESCRIPTIONS[label_b]}\n"
        f"Mechanisms: {mechs_b_str}\n"
        f"Training examples of Level {label_b}:\n{ex_b_text}\n\n"
        f"Which level better fits the target utterance: "
        f"Level {label_a} ({DEFENSE_LEVELS[label_a]}) or "
        f"Level {label_b} ({DEFENSE_LEVELS[label_b]})?\n\n"
        "reasoning: <one sentence>\n"
        "label: <0-8>"
    )


# ---------------------------------------------------------------------------
# Phase 3: Deliberation Moderator (fallback when pairwise is not applicable)
# ---------------------------------------------------------------------------

DELIBERATION_SYSTEM = f"""\
You are the Chief Arbiter of a multi-phase clinical assessment panel.

You will receive:
1. Phase 1: Independent analyst assessments with labels, identified \
   mechanisms, confidence, and alternative labels.
2. Phase 2: Targeted class-advocate evaluations with fit ratings (STRONG / \
   MODERATE / WEAK) for each candidate label, based on comparison with \
   real training examples of that class.

Your decision process:
1. Note which labels were proposed by the Phase 1 analysts and how many support each.
2. Review the Phase 2 advocate evidence — this is critical for rare classes.
3. Apply these rules in order:
   a. If one candidate has STRONG advocate fit and others have WEAK → choose it.
   b. If multiple candidates have STRONG fit → prefer the one with more Phase 1 support.
   c. If fits are tied (e.g. both MODERATE) → prefer the label with more Phase 1 votes \
      and higher confidence.
   d. If Phase 1 is unanimous but the advocate for that label is WEAK → still \
      respect the unanimous assessment (advocates can be wrong too).
4. IMPORTANT: Level 7 comprises ~52% of training data. Models systematically \
   over-predict it. Before confirming L7, verify the speaker is ACTIVELY coping \
   with a stressor (not just narrating, agreeing, or expressing emotion). If \
   uncertain between L7 and a lower level, prefer the lower level per DMRS \
   coding handbook rules.
5. If multiple defenses are plausible, the lower (less mature) level is typically \
   the primary label per DMRS coding rules.
{_HANDBOOK_RULES}
End with exactly:
reasoning: <one-sentence synthesis>
label: <0-8>
"""


def _build_deliberation_prompt(
    sample: dict,
    verdicts: list["AgentVerdict"],
    advocate_results: list["AdvocateResult"],
) -> str:
    dialogue = format_dialogue(sample)

    # Phase 1 summary
    verdict_lines = []
    for v in verdicts:
        lname = DEFENSE_LEVELS.get(v.label, "?")
        alt_name = DEFENSE_LEVELS.get(v.alternative_label, "?")
        verdict_lines.append(
            f"— {v.agent_name}: Level {v.label} ({lname}) "
            f"[confidence={v.confidence}, mechanism={v.mechanism}, "
            f"alternative=Level {v.alternative_label} ({alt_name})]"
        )
    verdict_text = "\n".join(verdict_lines)

    # Phase 2 summary
    advocate_lines = []
    for a in advocate_results:
        aname = DEFENSE_LEVELS.get(a.label, "?")
        advocate_lines.append(
            f"— Level {a.label} ({aname}): fit={a.fit}  |  {a.evidence_summary}"
        )
    advocate_text = "\n".join(advocate_lines) if advocate_lines else "(no advocate evidence)"

    return (
        f"Dialogue:\n{dialogue}\n\n"
        f"Target utterance (by {sample['dialogue'][-1]['speaker']}):\n"
        f"{sample['current_text']}\n\n"
        f"DMRS Level Reference:\n{LABEL_SUMMARY}\n\n"
        f"=== Phase 1: Analyst Assessments ===\n{verdict_text}\n\n"
        f"=== Phase 2: Class Advocate Evidence ===\n{advocate_text}\n\n"
        "Synthesise all evidence and determine the final defense level.\n"
        "reasoning: <one sentence>\n"
        "label: <0-8>"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentVerdict:
    """Structured output from a Phase 1 specialist agent."""
    agent_name: str
    label: int
    confidence: str              # "high", "medium", "low"
    mechanism: str = "unknown"
    alternative_label: int = -1
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class AdvocateResult:
    """Structured output from a Phase 2 class advocate."""
    label: int
    fit: str = "WEAK"            # "STRONG", "MODERATE", "WEAK"
    evidence_summary: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_verdict(raw: str, agent_name: str) -> AgentVerdict:
    """
    Extract label, confidence, mechanism, and alternative from an agent response.
    """
    label = -1
    confidence = "medium"
    mechanism = "unknown"
    alternative = -1
    reasoning = raw.strip()

    label_match = re.search(r"label\s*:\s*(\d)", raw, re.IGNORECASE)
    if label_match:
        label = int(label_match.group(1))

    conf_match = re.search(r"confidence\s*:\s*(high|medium|low)", raw, re.IGNORECASE)
    if conf_match:
        confidence = conf_match.group(1).lower()

    mech_match = re.search(r"mechanism\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if mech_match:
        mechanism = mech_match.group(1).strip()

    alt_match = re.search(r"alternative\s*:\s*(\d)", raw, re.IGNORECASE)
    if alt_match:
        alternative = int(alt_match.group(1))

    return AgentVerdict(
        agent_name=agent_name,
        label=label,
        confidence=confidence,
        mechanism=mechanism,
        alternative_label=alternative,
        reasoning=reasoning,
        raw_response=raw,
    )


def _parse_advocate(raw: str, label: int) -> AdvocateResult:
    """Extract fit rating and evidence summary from a class advocate response."""
    fit = "WEAK"
    evidence_summary = ""

    fit_match = re.search(r"fit\s*:\s*(STRONG|MODERATE|WEAK)", raw, re.IGNORECASE)
    if fit_match:
        fit = fit_match.group(1).upper()

    ev_match = re.search(r"evidence_summary\s*:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if ev_match:
        evidence_summary = ev_match.group(1).strip()

    return AdvocateResult(
        label=label,
        fit=fit,
        evidence_summary=evidence_summary,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def _call_google(
    client: genai.Client,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Call the Google Generative AI (Gemini) API."""
    # Thinking models (2.5-pro, 2.5-flash) use internal reasoning tokens
    # that count against max_output_tokens — give them headroom.
    effective_max = max(max_tokens, 8192) if "2.5" in model else max_tokens
    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=effective_max,
    )
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=model,
            contents=user,
            config=config,
        ),
        timeout=120,
    )
    return response.text or ""


# ---------------------------------------------------------------------------
# Phase 1 prompt builders
# ---------------------------------------------------------------------------

def _user_prompt_base(sample: dict) -> str:
    """Common dialogue + target block used in every agent prompt."""
    conversation = format_dialogue(sample)
    return (
        f"Dialogue context:\n{conversation}\n\n"
        f"Target utterance (by {sample['dialogue'][-1]['speaker']}):\n"
        f"{sample['current_text']}\n"
    )


def _agent_output_instructions() -> str:
    return (
        "\n\nProvide your analysis, then end with exactly these 4 lines:\n"
        "mechanism: <specific DMRS mechanism name, or \"None\">\n"
        "confidence: <high|medium|low>\n"
        "label: <0-8>\n"
        "alternative: <0-8>\n"
        "No additional content after the alternative line."
    )


def build_clinical_prompt(sample: dict) -> str:
    return (
        f"DMRS Level Reference:\n{LEVEL_REFERENCE}\n\n"
        + _user_prompt_base(sample)
        + _agent_output_instructions()
    )


def build_mechanism_prompt(sample: dict) -> str:
    return (
        f"DMRS Level Reference:\n{LEVEL_REFERENCE}\n\n"
        + _user_prompt_base(sample)
        + _agent_output_instructions()
    )


def build_pattern_prompt(sample: dict, few_shot_examples: list[dict]) -> str:
    examples_text = format_few_shot_examples(few_shot_examples)
    return (
        f"DMRS Level Reference:\n{LABEL_SUMMARY}\n\n"
        f"=== Labelled examples for reference ===\n{examples_text}\n"
        f"=== New dialogue to classify ===\n"
        + _user_prompt_base(sample)
        + _agent_output_instructions()
    )


# ---------------------------------------------------------------------------
# Council orchestrator
# ---------------------------------------------------------------------------

class Council:
    """
    Multi-phase deliberative council for DMRS defense-level classification.

    Phase 1: 3 agents assess independently (parallel, 3 calls).
    Phase 2: Class advocates evaluate candidate labels with class-specific
             training examples (conditional — skipped on consensus, 2-5 calls).
    Phase 3: Smart resolution using advocate fit ratings, with optional
             pairwise head-to-head for ambiguous cases (0-1 calls).

    Usage::

        council = Council(config, retriever)
        prediction = await council.predict(sample)
    """

    _ALL_PHASE1_AGENTS = [
        ("Clinical Analyst",     CLINICAL_ANALYST_SYSTEM,     build_clinical_prompt),
        ("Mechanism Specialist", MECHANISM_SPECIALIST_SYSTEM, build_mechanism_prompt),
        ("Pattern Analyst",      PATTERN_ANALYST_SYSTEM,      None),  # needs few-shot
    ]

    def __init__(
        self,
        config: CouncilConfig,
        retriever: ExampleRetriever | None = None,
        exclude_agents: set[str] | None = None,
        excluded_labels: set[int] | None = None,
    ):
        self.config = config
        self.retriever = retriever
        self.excluded_labels = excluded_labels or set()
        self._clients: dict[str, genai.Client] = {}

        skip = set(exclude_agents or set())
        self.phase1_agents = [
            (name, sys_prompt, pfn)
            for name, sys_prompt, pfn in self._ALL_PHASE1_AGENTS
            if name not in skip
        ]

    # ---- client management ------------------------------------------------

    def _get_client(self, model_cfg: ModelConfig) -> genai.Client:
        """Cache one client per API key."""
        key = model_cfg.api_key or ""
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=model_cfg.api_key)
        return self._clients[key]

    def _model_for(self, agent_name: str) -> ModelConfig:
        return self.config.agent_overrides.get(agent_name, self.config.default_model)

    def _moderator_model(self) -> ModelConfig:
        return self.config.moderator_model or self.config.default_model

    # ---- single LLM call --------------------------------------------------

    async def _call(
        self,
        model_cfg: ModelConfig,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Call the Gemini API."""
        client = self._get_client(model_cfg)
        temp = temperature if temperature is not None else model_cfg.temperature
        mtok = max_tokens if max_tokens is not None else model_cfg.max_tokens
        return await _call_google(client, model_cfg.model, system, user, temp, mtok)

    # ---- Phase 1: Initial assessment --------------------------------------

    async def _run_agent(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AgentVerdict:
        """Run one specialist agent and parse its verdict."""
        mcfg = self._model_for(agent_name)
        try:
            raw = await self._call(mcfg, system_prompt, user_prompt)
        except Exception as e:
            logger.warning("Agent %s failed: %s", agent_name, e)
            raw = f"ERROR: {e}"
        return _parse_verdict(raw, agent_name)

    async def _phase1(self, sample: dict) -> list[AgentVerdict]:
        """Run all Phase 1 agents in parallel."""
        few_shot: list[dict] = []
        if self.retriever is not None:
            few_shot = self.retriever.retrieve(
                sample, k=3,
                exclude_dialogue_id=sample.get("dialogue_id"),
                exclude_labels=self.excluded_labels or None,
            )

        tasks: list[tuple[str, str, str]] = []
        for name, system, prompt_fn in self.phase1_agents:
            if name == "Pattern Analyst":
                user_prompt = build_pattern_prompt(sample, few_shot)
            else:
                assert prompt_fn is not None
                user_prompt = prompt_fn(sample)
            tasks.append((name, system, user_prompt))

        sem = asyncio.Semaphore(self.config.max_parallel_agents)

        async def _guarded(n: str, s: str, u: str) -> AgentVerdict:
            async with sem:
                return await self._run_agent(n, s, u)

        return list(await asyncio.gather(
            *[_guarded(n, s, u) for n, s, u in tasks]
        ))

    # ---- Phase 2: Class advocates -----------------------------------------

    async def _run_advocate(
        self,
        sample: dict,
        label: int,
        class_examples: list[dict],
    ) -> AdvocateResult:
        """Run one class advocate for a specific label."""
        system = _build_advocate_system(label)
        user = _build_advocate_prompt(sample, label, class_examples)
        mcfg = self._model_for("advocate")
        try:
            raw = await self._call(
                mcfg, system, user,
                temperature=0.2,
                max_tokens=512,
            )
        except Exception as e:
            logger.warning("Advocate for Level %d failed: %s", label, e)
            raw = f"ERROR: {e}"
        return _parse_advocate(raw, label)

    async def _phase2(
        self,
        sample: dict,
        candidate_labels: set[int],
    ) -> list[AdvocateResult]:
        """Run class advocates for all candidate labels in parallel."""
        if self.retriever is None:
            return []

        sem = asyncio.Semaphore(self.config.max_parallel_agents)

        async def _guarded(label: int) -> AdvocateResult:
            async with sem:
                # Give minority classes more exemplars so advocates see
                # greater within-class diversity.
                class_size = len(self.retriever.label_to_indices.get(label, []))
                k = 5 if class_size < 100 else 3
                examples = self.retriever.retrieve_for_class(
                    sample, label, k=k,
                    exclude_dialogue_id=sample.get("dialogue_id"),
                )
                return await self._run_advocate(sample, label, examples)

        return list(await asyncio.gather(
            *[_guarded(lbl) for lbl in sorted(candidate_labels)]
        ))

    # ---- Phase 3: Resolution ----------------------------------------------

    async def _resolve(
        self,
        sample: dict,
        verdicts: list[AgentVerdict],
        advocates: list[AdvocateResult],
    ) -> tuple[int, str]:
        """
        Resolve the final label using advocate fit ratings.

        Strategy:
          1. If exactly one candidate has STRONG fit → pick it immediately.
          2. If multiple STRONG or all MODERATE → pairwise head-to-head.
          3. If all WEAK or no advocates → fall back to deliberation moderator.
        """
        fit_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1}
        strong = [a for a in advocates if a.fit == "STRONG"]
        moderate = [a for a in advocates if a.fit == "MODERATE"]

        # Case 1: Exactly one STRONG → clear winner
        if len(strong) == 1:
            winner = strong[0].label
            logger.debug("  Resolve: single STRONG → L%d", winner)
            return winner, f"STRONG advocate evidence for L{winner}"

        # Case 2: Multiple STRONG → pairwise between them
        if len(strong) >= 2:
            # Pick the two with most Phase 1 support
            support = self._label_support(verdicts)
            sorted_strong = sorted(
                strong, key=lambda a: support.get(a.label, 0), reverse=True,
            )
            label_a, label_b = sorted_strong[0].label, sorted_strong[1].label
            logger.debug("  Resolve: multi-STRONG pairwise L%d vs L%d", label_a, label_b)
            final = await self._pairwise_resolve(sample, label_a, label_b)
            return final, f"pairwise between STRONG L{label_a} and L{label_b}"

        # Case 3: No STRONG but have MODERATE → pairwise between top-2 MODERATE
        if len(moderate) >= 2:
            support = self._label_support(verdicts)
            sorted_mod = sorted(
                moderate, key=lambda a: support.get(a.label, 0), reverse=True,
            )
            label_a, label_b = sorted_mod[0].label, sorted_mod[1].label
            logger.debug("  Resolve: multi-MODERATE pairwise L%d vs L%d", label_a, label_b)
            final = await self._pairwise_resolve(sample, label_a, label_b)
            return final, f"pairwise between MODERATE L{label_a} and L{label_b}"

        # Case 4: One MODERATE and rest WEAK → pick the MODERATE one
        if len(moderate) == 1:
            winner = moderate[0].label
            logger.debug("  Resolve: single MODERATE → L%d", winner)
            return winner, f"only MODERATE advocate for L{winner}"

        # Case 5: All WEAK or no advocates → use deliberation moderator
        logger.debug("  Resolve: all WEAK → deliberation moderator")
        return await self._deliberate(sample, verdicts, advocates)

    async def _pairwise_resolve(
        self,
        sample: dict,
        label_a: int,
        label_b: int,
    ) -> int:
        """Head-to-head comparison of two candidates with class-specific examples."""
        examples_a: list[dict] = []
        examples_b: list[dict] = []
        if self.retriever is not None:
            did = sample.get("dialogue_id")
            examples_a = self.retriever.retrieve_for_class(
                sample, label_a, k=3, exclude_dialogue_id=did,
            )
            examples_b = self.retriever.retrieve_for_class(
                sample, label_b, k=3, exclude_dialogue_id=did,
            )

        prompt = _build_pairwise_prompt(
            sample, label_a, label_b, examples_a, examples_b,
        )
        mod_cfg = self._moderator_model()
        try:
            raw = await self._call(
                mod_cfg, PAIRWISE_SYSTEM, prompt,
                temperature=0.1,
                max_tokens=512,
            )
        except Exception as e:
            logger.warning("Pairwise resolve failed: %s", e)
            return label_a

        verdict = _parse_verdict(raw, "Pairwise")
        if verdict.label in (label_a, label_b):
            return verdict.label
        if 0 <= verdict.label <= 8:
            logger.info(
                "Pairwise returned L%d (not in {%d,%d}); using it",
                verdict.label, label_a, label_b,
            )
            return verdict.label
        return label_a  # fallback

    async def _deliberate(
        self,
        sample: dict,
        verdicts: list[AgentVerdict],
        advocates: list[AdvocateResult],
    ) -> tuple[int, str]:
        """Open-ended moderator deliberation (fallback)."""
        mod_cfg = self._moderator_model()
        prompt = _build_deliberation_prompt(sample, verdicts, advocates)
        try:
            raw = await self._call(
                mod_cfg, DELIBERATION_SYSTEM, prompt,
                temperature=0.1,
                max_tokens=512,
            )
        except Exception as e:
            logger.warning("Deliberation failed: %s", e)
            raw = ""

        verdict = _parse_verdict(raw, "Deliberation")
        if 0 <= verdict.label <= 8:
            return verdict.label, verdict.reasoning

        label = self._majority_vote(verdicts, advocates)
        logger.info("Deliberation parse failed; evidence-weighted vote → %d", label)
        return label, "fallback evidence-weighted vote"

    # ---- Helper: label support from Phase 1 verdicts ----------------------

    @staticmethod
    def _label_support(verdicts: list[AgentVerdict]) -> dict[int, float]:
        """Compute confidence-weighted support score for each label."""
        weight_map = {"high": 3, "medium": 2, "low": 1}
        scores: dict[int, float] = {}
        for v in verdicts:
            w = weight_map.get(v.confidence, 2)
            if 0 <= v.label <= 8:
                scores[v.label] = scores.get(v.label, 0) + w
            if 0 <= v.alternative_label <= 8:
                scores[v.alternative_label] = scores.get(
                    v.alternative_label, 0,
                ) + w * 0.3
        return scores

    @staticmethod
    def _majority_vote(
        verdicts: list[AgentVerdict],
        advocates: list[AdvocateResult] | None = None,
    ) -> int:
        """Weighted vote incorporating both Phase 1 verdicts and Phase 2
        advocate evidence.  Advocate fit ratings act as a multiplier so
        that a rare class with STRONG advocate evidence can outweigh
        majority-class votes with only MODERATE or WEAK support."""
        conf_weight = {"high": 3, "medium": 2, "low": 1}
        scores: dict[int, float] = {}
        for v in verdicts:
            if 0 <= v.label <= 8:
                w = conf_weight.get(v.confidence, 2)
                scores[v.label] = scores.get(v.label, 0) + w

        # Incorporate advocate fit as an additive bonus
        if advocates:
            fit_bonus = {"STRONG": 4, "MODERATE": 2, "WEAK": 0}
            for a in advocates:
                if 0 <= a.label <= 8:
                    scores[a.label] = scores.get(a.label, 0) + fit_bonus.get(a.fit, 0)

        if not scores:
            return 0
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    # ---- Minority class screening ----------------------------------------

    def _best_minority_candidate(
        self,
        sample: dict,
        already_candidates: set[int],
    ) -> int | None:
        """
        Return the minority class with highest retrieval similarity to the
        sample that is not already in the candidate set, or None.
        """
        if self.retriever is None:
            return None
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        query_vec = self.retriever.vectorizer.transform(
            [format_dialogue(sample)]
        )
        best_label: int | None = None
        best_score: float = -1.0
        for mlabel in self.retriever.minority_labels:
            if mlabel in already_candidates:
                continue
            class_indices = self.retriever.label_to_indices[mlabel]
            class_matrix = self.retriever.tfidf_matrix[class_indices]
            sims = cos_sim(query_vec, class_matrix).ravel()
            top_sim = float(sims.max())
            if top_sim > best_score:
                best_score = top_sim
                best_label = mlabel
        return best_label

    # ---- Main prediction pipeline -----------------------------------------

    async def predict(self, sample: dict) -> dict[str, Any]:
        """
        Run the full multi-phase council pipeline for one sample.

        Returns dict with keys: ``id``, ``label``, ``phase``, ``verdicts``,
        ``advocates``, ``deliberation``.
        """
        # ── Phase 1: Independent assessment ────────────────────────────────
        verdicts = await self._phase1(sample)

        for v in verdicts:
            logger.debug(
                "  P1 %s → L%d [%s] mech=%s alt=L%d",
                v.agent_name, v.label, v.confidence, v.mechanism, v.alternative_label,
            )

        # ── Consensus check ───────────────────────────────────────────────
        valid_labels = [v.label for v in verdicts if 0 <= v.label <= 8]
        unique_labels = set(valid_labels)
        high_conf = sum(1 for v in verdicts if v.confidence == "high")

        if len(unique_labels) == 1 and high_conf >= len(verdicts) and valid_labels[0] == 0:
            # Only auto-accept L0 consensus — all other labels need Phase 2
            # advocate verification to combat L7 over-prediction
            final_label = valid_labels[0]
            logger.debug("  Consensus → L%d (3 calls)", final_label)
            return {
                "id": sample["id"],
                "label": final_label,
                "phase": "consensus",
                "verdicts": [self._verdict_dict(v) for v in verdicts],
                "advocates": [],
                "deliberation": {
                    "label": final_label,
                    "reasoning": "unanimous high-confidence consensus",
                },
            }

        # ── Phase 2: Differential diagnosis ───────────────────────────────
        candidates: set[int] = set()
        for v in verdicts:
            if 0 <= v.label <= 8 and v.label not in self.excluded_labels:
                candidates.add(v.label)
            if 0 <= v.alternative_label <= 8 and v.alternative_label not in self.excluded_labels:
                candidates.add(v.alternative_label)

        # Ensure at least 2 candidates for meaningful comparison
        if len(candidates) < 2 and valid_labels:
            for fallback in [7, 0, 6, 3]:
                if fallback not in candidates and fallback not in self.excluded_labels:
                    candidates.add(fallback)
                    break

        # ── Minority class screening ─────────────────────────────────────
        # Always evaluate the best-matching minority class even if no
        # Phase 1 agent proposed it.  This is cheap (1 extra advocate call)
        # and catches cases where all agents defaulted to a majority class.
        if self.retriever is not None:
            minority_candidate = self._best_minority_candidate(
                sample, candidates,
            )
            if minority_candidate is not None:
                candidates.add(minority_candidate)

        logger.debug("  Phase2 candidates: %s", sorted(candidates))
        advocates = await self._phase2(sample, candidates)

        for a in advocates:
            logger.debug(
                "  Advocate L%d → fit=%s: %s",
                a.label, a.fit, a.evidence_summary[:80] if a.evidence_summary else "",
            )

        # ── Phase 3: Smart resolution ─────────────────────────────────────
        final_label, reasoning = await self._resolve(sample, verdicts, advocates)
        logger.debug("  Final → L%d (%s)", final_label, reasoning[:80] if reasoning else "")

        return {
            "id": sample["id"],
            "label": final_label,
            "phase": "resolved",
            "verdicts": [self._verdict_dict(v) for v in verdicts],
            "advocates": [self._advocate_dict(a) for a in advocates],
            "deliberation": {"label": final_label, "reasoning": reasoning},
        }

    # ---- serialisation helpers --------------------------------------------

    @staticmethod
    def _verdict_dict(v: AgentVerdict) -> dict:
        return {
            "agent": v.agent_name,
            "label": v.label,
            "confidence": v.confidence,
            "mechanism": v.mechanism,
            "alternative": v.alternative_label,
            "reasoning": v.reasoning,
        }

    @staticmethod
    def _advocate_dict(a: AdvocateResult) -> dict:
        return {
            "label": a.label,
            "fit": a.fit,
            "evidence_summary": a.evidence_summary,
        }

    # ---- batch prediction -------------------------------------------------

    async def predict_batch(
        self,
        samples: list[dict],
        max_concurrent: int = 5,
    ) -> list[dict]:
        """Run the council on a batch of samples with concurrency control."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _process(s: dict) -> dict:
            async with sem:
                return await self.predict(s)

        return list(await asyncio.gather(*[_process(s) for s in samples]))
