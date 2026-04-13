"""
Configuration and constants for the DMRS Council of LLMs classifier.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Label taxonomy (Defense Mechanism Rating Scales)
# ---------------------------------------------------------------------------

DEFENSE_LEVELS: dict[int, str] = {
    0: "No Defense",
    1: "Action Defense Level",
    2: "Major Image-Distorting Defense Level",
    3: "Disavowal Defense Level",
    4: "Minor Image-Distorting Defense Level",
    5: "Neurotic Defense Level",
    6: "Obsessional Defense Level",
    7: "Highly Adaptive Defense Level",
    8: "Needs More Information",
}

DEFENSE_MECHANISMS: dict[int, list[str]] = {
    0: [],
    1: ["Acting Out", "Help-Rejecting Complaining", "Passive Aggression"],
    2: ["Splitting", "Projective Identification"],
    3: ["Denial", "Projection", "Rationalization", "Autistic Fantasy"],
    4: ["Devaluation", "Idealization", "Omnipotence"],
    5: ["Displacement", "Dissociation", "Reaction Formation", "Repression"],
    6: ["Intellectualization", "Isolation of Affects", "Undoing"],
    7: [
        "Affiliation", "Altruism", "Anticipation", "Humor",
        "Self-Assertion", "Self-Observation", "Sublimation", "Suppression",
    ],
    8: [],
}

DEFENSE_DESCRIPTIONS: dict[int, str] = {
    0: (
        "Utterances that serve a purely conversational or social function "
        "without engaging with psychological conflict or coping. Includes: "
        "greetings, farewells, minimal backchannels ('okay', 'I see', 'yeah'), "
        "simple thanks or acknowledgments, logistical exchanges, and neutral "
        "small talk. The key criterion: the speaker is NOT processing, avoiding, "
        "or transforming any emotional content — the utterance is socially "
        "functional rather than psychologically functional."
    ),
    1: (
        "Dealing with internal conflicts by acting on the environment. "
        "Distress is channeled into behavior, often impulsively and without "
        "reflection, to release tension, gratify wishes, or avoid painful "
        "feelings."
    ),
    2: (
        "Coping with intolerable anxiety by grossly distorting the image of "
        "oneself or others. Splitting representations into polar opposites "
        "(all-good or all-bad) simplifies reality and protects from ambivalence."
    ),
    3: (
        "Refusing to acknowledge unacceptable aspects of reality or one's own "
        "experience. Justifying not taking responsibility by denying existence "
        "of a problem, providing excuses, attributing it to others, or "
        "retreating into fantasy."
    ),
    4: (
        "Protecting self-esteem from threats like failure or criticism by "
        "distorting one's image less severely than Level 2. Temporarily "
        "boosting self-esteem by attributing exaggerated positive or negative "
        "qualities to oneself or others."
    ),
    5: (
        "Managing emotional conflict by keeping unacceptable wishes, thoughts, "
        "or motives out of conscious awareness. Experiencing feelings while the "
        "idea is blocked (or vice versa), leading to indirect or displaced "
        "expressions."
    ),
    6: (
        "Managing threatening feelings by separating them from the thoughts or "
        "events that caused them. Remaining aware of cognitive details but "
        "avoiding emotional impact through excessive logic, abstract thinking, "
        "or symbolic acts."
    ),
    7: (
        "The most adaptive and constructive ways of handling stressors. "
        "The speaker engages constructively with psychological conflict "
        "rather than avoiding or distorting it. Includes: seeking or accepting "
        "support (affiliation), expressing needs or boundaries (self-assertion), "
        "reflecting on one's own feelings or patterns (self-observation), "
        "planning ahead (anticipation), consciously setting aside distress "
        "to function (suppression), using humor, helping others (altruism), "
        "or channeling emotions into productive activity (sublimation)."
    ),
    8: (
        "The utterance is too ambiguous or the context is insufficient to "
        "determine a defense level. Evidence triggers suspicion of a defense "
        "but is insufficient to confirm any tier."
    ),
}

# ---------------------------------------------------------------------------
# DMRS-Q behavioral items (curated: 2-3 most discriminative per mechanism)
# ---------------------------------------------------------------------------

DMRS_Q_ITEMS: dict[int, dict[str, list[str]]] = {
    0: {"No Defense": [
        "Greetings, farewells, simple thanks, yes/no/okay, factual questions",
        "Social small talk, reporting factual details, acknowledging therapist",
    ]},
    1: {
        "Passive Aggression": [
            "Fails to express opposition directly; uses indirect, annoying ways (e.g., silence, procrastination)",
            "Outwardly cooperative but procrastinates and refuses to comply",
        ],
        "Help-Rejecting Complaining": [
            "Recites litany of problems but not engaged in solving them; prefers to complain",
            "Treats problems as insoluble, systematically rejects others' suggestions",
        ],
        "Acting Out": [
            "Acts impulsively on interpersonal disappointment without reflection on consequences",
            "Resorts to uncontrolled behaviors (binge-eating, reckless driving, etc.) as escape from distress",
        ],
    },
    2: {
        "Splitting": [
            "Has periods of saying highly positive things about self/others, then highly negative, without noticing the contradiction",
            "Experiences others in 'black or white' terms, failing to form balanced, realistic views",
        ],
        "Projective Identification": [
            "Gets angry at someone for no apparent reason, then accuses them of intending to provoke",
            "Assumes other's feelings are the same as own; tends to 'put words in the other's mouth'",
        ],
    },
    3: {
        "Denial": [
            "When confronted with meaningful topics, denies they are important and refuses to discuss",
            "Contrary to evidence, claims to have done something they did not do; becomes irritated if confronted",
        ],
        "Rationalization": [
            "Makes excuses or points out others' contributions to avoid taking responsibility for own actions",
            "Avoids acknowledging own feelings by giving a plausible but incorrect explanation that covers real reasons",
        ],
        "Projection": [
            "When confronted about own feelings/intentions, is evasive but talks about similar feelings in others",
            "Perceives others as untrustworthy or manipulative when there is no objective basis for these concerns",
        ],
        "Autistic Fantasy": [
            "Has repetitive daydreams in lieu of real-life social relationships",
            "Prefers to daydream about solutions rather than planning direct, realistic actions",
        ],
    },
    4: {
        "Devaluation": [
            "Says demeaning things about self ('I am so stupid') or dismisses others' accomplishments",
            "When experiencing failure or shame, dismisses the issue by saying something negative about self, then moves to another topic",
        ],
        "Idealization": [
            "Makes many references to how important self/others are, emphasizing image rather than real accomplishments",
            "When confronted with problems, dwells on positive qualities ('I'm the best') as if that solves things",
        ],
        "Omnipotence": [
            "Acts in a very self-assured 'I can handle anything' way in the face of problems they cannot fully control",
            "Excessive bravado or grandiosity in describing personal plans or accomplishments",
        ],
    },
    5: {
        "Repression": [
            "Keeps unpleasant things vague; has trouble remembering or can't recall specific examples",
            "When a topic is emotionally loaded, forgets what they were talking about and gets lost",
        ],
        "Dissociation": [
            "Behaves in a very uncharacteristic way that seems out of usual control; is surprised by it",
            "In response to emotional situation, becomes confused, depersonalized, 'spaced out', or can't think",
        ],
        "Reaction Formation": [
            "When confronting a personal wish, substitutes an opposite attitude (e.g., desire → renunciation)",
            "In dealing with angry/abusive people, is cooperative and nice, failing to express any expected negative feelings",
        ],
        "Displacement": [
            "When dealing with an important anxiety-provoking problem, prefers to focus on minor or unrelated matters",
            "Directs strong feelings toward a person or object with little connection to the actual source",
        ],
    },
    6: {
        "Isolation of Affect": [
            "When telling an emotionally meaningful story, states they have no feelings about it, though they recognize they should",
            "Describes distressing experiences clearly but without any attendant emotion",
        ],
        "Intellectualization": [
            "When confronting personal issues, asks general questions as if getting abstract information will elucidate own feelings",
            "Talks about personal experiences by making general statements that avoid revealing specific personal feelings",
        ],
        "Undoing": [
            "Prefaces a strong statement with a disclaimer that what they are about to say may not be true",
            "Conveys opinions with a series of opposite or contradictory statements, uncomfortable taking a clear stand",
        ],
    },
    7: {
        "Affiliation": [
            "When bringing a problem to someone, not expecting them to fix it, but to collaboratively find a solution they will implement",
            "Describes how talking to others helped them think through and handle a problem",
        ],
        "Altruism": [
            "Helps others with a problem that has personal meaning related to their own past experiences",
            "Participates in helping others in direct person-to-person ways, finding it rewarding",
        ],
        "Anticipation": [
            "Ahead of an important event, practices imagining the situation to be better prepared and less anxious",
            "Characteristically mentions thinking about outcomes ahead of time and emotionally preparing",
        ],
        "Humor": [
            "Makes amusing or ironic comments about embarrassing situations to diffuse them",
            "In confronting difficult situations, uses humor to mitigate negative feelings without hostility",
        ],
        "Self-Assertion": [
            "When someone is impolite or dismissive, stands up appropriately even if they cannot change the outcome",
            "When confronted with emotionally difficult situations, expresses thoughts, wishes, or feelings clearly and directly",
        ],
        "Self-Observation": [
            "When talking about a personally charged topic, displays an accurate view of self and can see from others' perspective",
            "When confronting emotionally important problems, reflects on relevant personal experiences and explores emotional reactions",
        ],
        "Sublimation": [
            "In describing artistic or creative activities, the process appears to transform emotional conflicts from elsewhere in life",
            "Following emotional distress, engages in sports or physical activities as an invigorating outlet",
        ],
        "Suppression": [
            "When presented with an external demanding situation, can put negative feelings aside to deal with what must be done",
            "When experiencing a desire that would have bad consequences, consciously decides to put it aside and not act on it",
        ],
    },
    8: {"Needs More Information": [
        "The utterance is too ambiguous or context is insufficient to determine function",
        "Evidence suggests a defense but is insufficient to confirm any specific tier",
    ]},
}


LABEL_SUMMARY = "\n".join(
    f"{k} = {DEFENSE_LEVELS[k]}"
    + (f" ({', '.join(DEFENSE_MECHANISMS[k])})" if DEFENSE_MECHANISMS[k] else "")
    for k in sorted(DEFENSE_LEVELS)
)


def get_label_summary(exclude: set[int] | None = None) -> str:
    """Generate LABEL_SUMMARY with specified labels excluded."""
    if not exclude:
        return LABEL_SUMMARY
    return "\n".join(
        f"{k} = {DEFENSE_LEVELS[k]}"
        + (f" ({', '.join(DEFENSE_MECHANISMS[k])})" if DEFENSE_MECHANISMS[k] else "")
        for k in sorted(DEFENSE_LEVELS)
        if k not in exclude
    )

# ---------------------------------------------------------------------------
# Model / API configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a Gemini model endpoint."""
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    api_key: str = "CHANGE_ME"
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class CouncilConfig:
    """Top-level configuration for the Council of LLMs."""

    # Each specialist agent can optionally use a different model config.
    # By default every agent shares the same config.
    default_model: ModelConfig = field(default_factory=ModelConfig)

    # Override per agent (keys must match agent names in AGENT_REGISTRY).
    agent_overrides: dict[str, ModelConfig] = field(default_factory=dict)

    # Moderator can use a stronger model for final synthesis.
    moderator_model: ModelConfig | None = None  # falls back to default_model

    # Few-shot retrieval
    num_few_shot: int = 5          # examples per label-diverse retrieval
    num_candidates: int = 30       # TF-IDF candidates before MMR re-ranking

    # Execution
    max_parallel_agents: int = 5   # how many agents to run concurrently
