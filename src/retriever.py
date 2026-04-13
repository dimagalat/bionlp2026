"""
Few-shot example retriever using TF-IDF similarity with robust selection.

Retrieval strategies
--------------------
* **General retrieval** (`retrieve`):  MMR (Maximal Marginal Relevance) for a
  principled balance between query-relevance and inter-example diversity, with
  label-coverage enforcement so the model sees varied defense levels.

* **Class-specific retrieval** (`retrieve_for_class`):  combines query
  similarity with *prototypicality* — how close an example sits to its own
  class centroid — so advocate agents receive exemplars that are both relevant
  to the query AND unambiguously representative of their class.

Both strategies are fully deterministic (no randomness), which eliminates one
source of run-to-run variability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _flatten_dialogue(sample: dict) -> str:
    """Concatenate all dialogue turns into a single string.

    Note: full-dialogue TF-IDF outperforms focused (last 3 turns) and
    semantic retrieval on this task. The diluted signal from early turns
    (greetings, logistics) helps MMR enforce label diversity by matching
    with L0 examples, preventing the L7-attraction bias that plagues
    more focused retrieval methods.
    """
    parts = []
    for turn in sample["dialogue"]:
        parts.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(parts)


class ExampleRetriever:
    """TF-IDF based retriever with prototypicality scoring and MMR selection."""

    # Classes with < this fraction of training data get minority anchors
    # injected into the candidate pool during general retrieval.
    MINORITY_THRESHOLD: float = 0.10  # 10% of training set

    def __init__(
        self,
        train_data: list[dict],
        num_candidates: int = 30,
        proto_weight: float = 0.35,
        mmr_lambda: float = 0.7,
        semantic_embeddings: np.ndarray | None = None,
        embed_fn=None,
    ):
        """
        Parameters
        ----------
        train_data : list[dict]
            Training examples, each with ``dialogue``, ``current_text``, ``label``.
        num_candidates : int
            Size of the TF-IDF shortlist before MMR re-ranking.
        proto_weight : float  (0–1)
            Weight of prototypicality vs. query-similarity in class retrieval.
            ``score = (1 - proto_weight) * sim(query, ex)
                    + proto_weight * sim(ex, class_centroid)``
        mmr_lambda : float  (0–1)
            MMR trade-off.  1.0 = pure relevance, 0.0 = pure diversity.
        semantic_embeddings : np.ndarray, optional
            Pre-computed semantic embeddings for train_data (n_samples, embed_dim).
            When provided, retrieval uses semantic similarity instead of TF-IDF.
        embed_fn : callable, optional
            Function that takes a list of strings and returns np.ndarray of embeddings.
            Required for semantic retrieval of query samples.
        """
        self.train_data = train_data
        self.num_candidates = num_candidates
        self.proto_weight = proto_weight
        self.mmr_lambda = mmr_lambda
        self._semantic_embeddings = semantic_embeddings
        self._embed_fn = embed_fn

        # ── Build TF-IDF index ────────────────────────────────────────────
        corpus = [_flatten_dialogue(s) for s in train_data]
        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # ── Index by label ────────────────────────────────────────────────
        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(train_data):
            self.label_to_indices[s["label"]].append(i)

        # ── Index by dialogue_id (for leakage prevention) ────────────────
        self.dialogue_id_to_indices: dict[str, set[int]] = defaultdict(set)
        self._id_to_index: dict[str, int] = {}
        for i, s in enumerate(train_data):
            if "dialogue_id" in s:
                self.dialogue_id_to_indices[s["dialogue_id"]].add(i)
            if "id" in s:
                self._id_to_index[s["id"]] = i

        # ── Precompute class centroids & prototypicality scores ───────────
        # centroid_k = mean TF-IDF vector for class k
        # proto[i]   = cosine(example_i, centroid_{label_i})
        self._centroids: dict[int, np.ndarray] = {}
        self._proto: np.ndarray = np.zeros(len(train_data), dtype=np.float64)

        for label, indices in self.label_to_indices.items():
            class_matrix = self.tfidf_matrix[indices]
            centroid = class_matrix.mean(axis=0)          # dense (1, F)
            centroid = np.asarray(centroid).ravel()
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._centroids[label] = centroid

            # Per-example prototypicality = cosine(example, centroid)
            sims = cosine_similarity(class_matrix, centroid.reshape(1, -1)).ravel()
            for rank, idx in enumerate(indices):
                self._proto[idx] = sims[rank]

        # ── Identify minority classes ────────────────────────────────────
        n_total = len(train_data)
        self.minority_labels: set[int] = {
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) / n_total < self.MINORITY_THRESHOLD
        }

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _make_query_text(self, sample: dict) -> str:
        """Build the text used for semantic embedding of a query."""
        prior = sample["dialogue"][-2]["text"][:80] if len(sample["dialogue"]) >= 2 else ""
        return f"Find the defense mechanism. Context: {prior} Utterance: {sample['current_text']}"

    def _semantic_similarities(self, query_sample: dict) -> np.ndarray | None:
        """Compute semantic similarity if embeddings are available."""
        if self._semantic_embeddings is not None and self._embed_fn is not None:
            query_text = self._make_query_text(query_sample)
            query_emb = self._embed_fn([query_text])
            return cosine_similarity(query_emb, self._semantic_embeddings).flatten()
        return None

    # ------------------------------------------------------------------
    # Leakage filtering
    # ------------------------------------------------------------------

    def _excluded_indices(
        self,
        query_sample: dict,
        exclude_dialogue_id: str | None = None,
    ) -> set[int]:
        """Return set of training indices to exclude for this query."""
        excluded: set[int] = set()
        # Exclude exact self-match
        sample_id = query_sample.get("id")
        if sample_id and sample_id in self._id_to_index:
            excluded.add(self._id_to_index[sample_id])
        # Exclude all samples from same dialogue
        did = exclude_dialogue_id or query_sample.get("dialogue_id")
        if did:
            excluded |= self.dialogue_id_to_indices.get(did, set())
        return excluded

    def _filter_indices(
        self,
        indices: np.ndarray,
        excluded: set[int],
    ) -> np.ndarray:
        """Remove excluded indices from an array."""
        if not excluded:
            return indices
        return np.array([i for i in indices if i not in excluded])

    # ------------------------------------------------------------------
    # MMR selection
    # ------------------------------------------------------------------

    def _mmr_select(
        self,
        query_vec: np.ndarray,
        candidate_indices: np.ndarray,
        k: int,
        *,
        enforce_label_diversity: bool = True,
    ) -> list[int]:
        """
        Maximal Marginal Relevance selection from a candidate pool.

        MMR(d) = λ · sim(d, query)  −  (1−λ) · max_{d' ∈ selected} sim(d, d')

        When *enforce_label_diversity* is True, the first pass picks at most
        one example per label (highest-MMR first), then a second pass fills
        remaining slots without the label constraint.
        """
        if len(candidate_indices) == 0:
            return []

        cand_matrix = self.tfidf_matrix[candidate_indices]
        query_sims = cosine_similarity(query_vec, cand_matrix).ravel()

        n_cand = len(candidate_indices)
        # Precompute pairwise similarities among candidates (dense, small)
        pair_sims = cosine_similarity(cand_matrix).astype(np.float64)

        selected_mask = np.zeros(n_cand, dtype=bool)
        selected_local: list[int] = []      # indices into candidate_indices
        covered_labels: set[int] = set()

        def _pick_next() -> int | None:
            """Return index (into candidate_indices) with highest MMR score."""
            best_idx = -1
            best_score = -np.inf
            for j in range(n_cand):
                if selected_mask[j]:
                    continue
                relevance = query_sims[j]
                if selected_local:
                    redundancy = pair_sims[j, selected_mask].max()
                else:
                    redundancy = 0.0
                score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = j
            return best_idx if best_idx >= 0 else None

        # Pass 1: label-diverse (one per label, highest-MMR first)
        if enforce_label_diversity:
            for _ in range(min(k, n_cand)):
                j = _pick_next()
                if j is None:
                    break
                real_idx = candidate_indices[j]
                label = self.train_data[real_idx]["label"]
                if label in covered_labels:
                    # Try others until we find an uncovered label or exhaust
                    # We do a small scan of top remaining MMR candidates
                    scores = []
                    for jj in range(n_cand):
                        if selected_mask[jj]:
                            continue
                        rl = self.train_data[candidate_indices[jj]]["label"]
                        if rl in covered_labels:
                            continue
                        relevance = query_sims[jj]
                        if selected_local:
                            redundancy = pair_sims[jj, selected_mask].max()
                        else:
                            redundancy = 0.0
                        s = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                        scores.append((jj, s))
                    if scores:
                        scores.sort(key=lambda x: x[1], reverse=True)
                        j = scores[0][0]
                        real_idx = candidate_indices[j]
                        label = self.train_data[real_idx]["label"]

                selected_mask[j] = True
                selected_local.append(j)
                covered_labels.add(label)
                if len(selected_local) >= k:
                    break

        # Pass 2: fill remaining slots (no label constraint, pure MMR)
        while len(selected_local) < k:
            j = _pick_next()
            if j is None:
                break
            selected_mask[j] = True
            selected_local.append(j)

        return [int(candidate_indices[j]) for j in selected_local]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_sample: dict,
        k: int = 3,
        exclude_dialogue_id: str | None = None,
        exclude_labels: set[int] | None = None,
    ) -> list[dict]:
        """
        Retrieve *k* diverse training examples for a query sample using MMR.

        Strategy:
          1. TF-IDF similarity → top *num_candidates* shortlist.
          2. Filter out same-dialogue and self-match examples (leakage prevention).
          3. Inject best-matching minority-class anchors into the pool so
             that MMR has the *opportunity* to select rare-class examples.
          4. MMR re-ranking with label-diversity enforcement.
        """
        query_vec = self.vectorizer.transform([_flatten_dialogue(query_sample)])
        tfidf_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        similarities = tfidf_sims

        # Leakage exclusion + label filtering
        excluded = self._excluded_indices(query_sample, exclude_dialogue_id)
        if exclude_labels:
            excluded |= {i for i, s in enumerate(self.train_data)
                         if s["label"] in exclude_labels}

        # Top candidates by similarity (over-fetch to compensate for filtering)
        fetch_n = self.num_candidates + len(excluded)
        top_indices = np.argsort(similarities)[::-1][:fetch_n]
        top_indices = self._filter_indices(top_indices, excluded)

        # ── Minority-class anchor injection ──────────────────────────────
        top_set = set(top_indices.tolist())
        labels_in_pool = {self.train_data[i]["label"] for i in top_indices}
        anchors: list[int] = []
        for mlabel in self.minority_labels:
            if mlabel in labels_in_pool:
                continue
            class_indices = [i for i in self.label_to_indices[mlabel]
                            if i not in excluded]
            if not class_indices:
                continue
            best_idx = max(class_indices, key=lambda i: similarities[i])
            if best_idx not in top_set:
                anchors.append(best_idx)
        if anchors:
            top_indices = np.concatenate([top_indices, np.array(anchors)])

        selected = self._mmr_select(
            query_vec, top_indices, k, enforce_label_diversity=True,
        )
        return [self.train_data[i] for i in selected]

    def retrieve_for_class(
        self,
        query_sample: dict,
        label: int,
        k: int = 3,
        exclude_dialogue_id: str | None = None,
    ) -> list[dict]:
        """
        Retrieve *k* training examples for a specific class, ranked by a
        blend of query-similarity and prototypicality.

        ``score = (1 − w) · sim(query, example)  +  w · proto(example)``

        Prototypicality ensures advocates see *clear*, representative
        exemplars rather than borderline cases that happen to match the query
        text.  This strongly reduces run-to-run variance in advocate judgments.
        """
        all_class_indices = self.label_to_indices.get(label, [])
        if not all_class_indices:
            return []

        # Leakage exclusion
        excluded = self._excluded_indices(query_sample, exclude_dialogue_id)
        class_indices = [i for i in all_class_indices if i not in excluded]
        if not class_indices:
            return []

        query_vec = self.vectorizer.transform([_flatten_dialogue(query_sample)])
        tfidf_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Use semantic similarity for within-class ranking when available,
        # blended with prototypicality. Semantic embeddings find better
        # functional matches within a class than TF-IDF surface matching.
        sem_sims = self._semantic_similarities(query_sample)
        q_sims = sem_sims if sem_sims is not None else tfidf_sims

        w = self.proto_weight
        scores = [
            ((1 - w) * q_sims[idx] + w * self._proto[idx], idx)
            for idx in class_indices
        ]
        scores.sort(key=lambda x: x[0], reverse=True)

        # Pick top-k by blended score, but also use MMR to avoid near-duplicate
        # examples within the class
        if k >= 3 and len(class_indices) >= k * 2:
            # Enough candidates to warrant MMR de-duplication
            top_class = np.array([idx for _, idx in scores[: k * 3]])
            selected = self._mmr_select(
                query_vec, top_class, k, enforce_label_diversity=False,
            )
            return [self.train_data[i] for i in selected]

        return [self.train_data[idx] for _, idx in scores[:k]]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        num_candidates: int = 30,
        proto_weight: float = 0.35,
        mmr_lambda: float = 0.7,
    ) -> "ExampleRetriever":
        data = json.load(open(path))
        return cls(
            data,
            num_candidates=num_candidates,
            proto_weight=proto_weight,
            mmr_lambda=mmr_lambda,
        )
