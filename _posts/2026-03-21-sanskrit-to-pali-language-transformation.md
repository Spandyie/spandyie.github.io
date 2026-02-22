---
layout: post
title: "Sanskrit → Pali: How Buddhism Transformed a Language"
date: 2026-03-21
tags: [sanskrit, pali, buddhism, linguistics, nlp, cross-lingual]
description: "When Buddhism spread from its Sanskrit-speaking origins, it chose Pali. Here's what that choice looks like in data — systematic sound changes, vocabulary that survived, and words that shifted meaning across traditions."
---

Every religion needs a language. Latin for medieval Catholicism, Classical Arabic for Islam, Sanskrit for the Brahminic tradition. When Buddhism emerged from northeastern India around the fifth century BCE, it made a choice: Pali, not Sanskrit.

That choice was not incidental. It was a philosophical statement — and it left a data trail we can follow.

---

## A Democratic Gesture

Sanskrit was the language of the Brahmin elite. To read it, you needed years of specialized training. Its complexity — the sandhi rules, the deep compounds, the eight cases — wasn't just an aesthetic feature; it was a barrier. Sacred knowledge in Sanskrit was necessarily restricted to those who could decode it.

The Buddha likely spoke Māgadhī, a Prakrit vernacular used in the Magadha region (roughly modern Bihar). He taught in the local language of wherever he was — not Sanskrit, not an elite register, but the spoken tongue. When his disciples began compiling the canon, they used Pali: a Prakrit closely related to Māgadhī, regularized for oral transmission.

This is a concrete sociological decision embedded in the language itself. Pali was chosen *because* it was more accessible, because it simplified the barriers Sanskrit erected. You can see this decision's fingerprints in the phonology.

---

## The Systematic Sound Changes

Pali isn't just "simplified Sanskrit." It's a systematic transformation — regular, predictable, reconstructable. Every Sanskrit word with a Pali cognate underwent the same set of changes. This is what linguists call a *sound correspondence*, and it's the signature of a language family relationship.

Here are the core transformations:

| Sanskrit | Pali | Pattern | Example |
|---------|------|---------|---------|
| *rm* | *mm* | Regressive assimilation | *karma* → *kamma* |
| *rv* | *bb* | Consonant simplification | *nirvāṇa* → *nibbāna* |
| *śr* → *sr* | *s* | Cluster reduction | *śrī* → *sirī* |
| *pt* | *tt* | Gemination | *sapta* (7) → *satta* |
| *ṣṭ* | *ṭṭh* | Retroflex simplification | *aṣṭa* (8) → *aṭṭha* |
| *ṣ* | *s* | Sibilant merger | *puruṣa* → *purisa* |
| *ā* + cluster | single consonant | Vowel lengthening + simplification | *rātri* → *ratti* |

These aren't exceptions or accidents. They're rules. Every Sanskrit word with a Pali cognate went through these same transformations. This regularity is what tells you Pali is a derivative of Sanskrit, not a separate invention — and it's what lets you reconstruct the etymology in both directions.

---

## Vocabulary That Survived, Vocabulary That Shifted

The interesting question isn't just what changed phonologically. It's what changed *semantically*. Same word (cognate), different meaning — a signature of cross-cultural transmission.

**Survived intact**: *samādhi* is the same in both traditions — meditative absorption. The technical terminology of meditation traveled across because the practice itself traveled across. The phenomenology was shared, so the vocabulary was shared.

**Shifted meaning — the big ones**:

*Dharma* (Sanskrit) → *dhamma* (Pali). In Vedic Sanskrit, *dharma* is cosmic order, the law that governs the universe, one's caste duty (*sva-dharma*). In the Pali canon, *dhamma* is the Buddha's teaching — the Path, the Truth, the way things are. The word survived; the center of gravity moved. The Buddhist tradition didn't invent a new term. It inherited a prestigious, already-meaningful word and filled it with new content.

*Ātman* (Sanskrit) → *attā* (Pali). In the Upanishads, *ātman* is the real Self — the eternal, unchanging core of personal identity that is ultimately identical to Brahman. In Pali Buddhism, *attā* appears primarily in the context of *anattā* — non-self, the teaching that there is no such fixed, eternal entity. The word survived to describe what Buddhism denies. The tradition preserved the vocabulary of the debate while reversing the conclusion.

*Brāhmaṇa* (Sanskrit) → *brāhmaṇa* (Pali, essentially unchanged). In Sanskrit tradition, a Brahmin is Brahmin by birth — caste is inherited. In the Dhammapada, the Buddha redefines *brāhmaṇa*: a true Brahmin is one who has purified their mind and achieved liberation, regardless of birth. Same word, inverted criterion. The Pali texts keep the high-status term and strip it of its hereditary basis.

**Lost entirely**: Much of the Vedic ritual vocabulary has no Pali equivalent because the rituals themselves didn't cross over. *Soma* (the ritual drink of the Rigveda), the names of specific Vedic sacrifices (*agniṣṭoma*, *aśvamedha*), the technical terms for fire altar construction — these terms appear in Sanskrit but have no corresponding Pali terms because the practices they named weren't part of the Buddhist tradition.

---

## Measuring Semantic Drift

The qualitative analysis above is supported by embedding-based quantification. Using IndicBERT (`ai4bharat/indic-bert`), we can embed Sanskrit and Pali terms in their respective contexts and measure cosine similarity:

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

# Embed in context to capture semantic usage, not just form
dharma_vedic = get_embedding("dharma cosmic law duty caste Brahmin Vedic ritual order")
dhamma_buddhist = get_embedding("dhamma teaching Buddha path suffering liberation nibbāna")

similarity = torch.cosine_similarity(dharma_vedic, dhamma_buddhist).item()
# Expected: ~0.60-0.70 (related, but meaningfully diverged)

atman_upanishad = get_embedding("ātman eternal self Brahman consciousness Upanishad identity")
anatta_buddhist = get_embedding("anattā non-self no fixed self impermanence dependent arising")

similarity_self = torch.cosine_similarity(atman_upanishad, anatta_buddhist).item()
# Expected: ~0.45-0.55 (lower — the traditions are in active disagreement here)
```

The embedding similarity scores reflect the degree of semantic overlap. High similarity: the concept traveled more or less intact. Low similarity: the word survived but the concept transformed. The *ātman/anattā* pair should show lower similarity than *dharma/dhamma* because Buddhism isn't just using *ātman* differently — it's arguing against the Upanishadic claim.

---

## The MITRA Corpus as Evidence

MITRA's 1.74 million Sanskrit/Pali-English parallel pairs (arXiv 2601.06400) provide a large-scale resource for this kind of cross-lingual analysis. By finding near-parallel passages — texts in both traditions covering similar topics — we can see which vocabulary transferred and which didn't.

Buddhist Sanskrit texts (Mahāyāna literature, Pali-adjacent but written in Sanskrit) provide a particularly interesting intermediate case: Sanskrit texts written by Buddhists who were drawing on the Pali tradition while using Sanskrit. The vocabulary choices in these texts illuminate the conscious decisions the tradition was making about terminology.

---

## A Personal Note

My family uses Sanskrit for ritual — prayers, ceremonies, the traditional forms. Pali appears occasionally in Buddhist contexts, in the chants I heard growing up.

I knew they were related. I didn't know how systematically. Until working through this analysis, I hadn't fully appreciated that when my family chants *kamma* in one context and the related concept would be *karma* in another, the difference in those two syllables encodes a thousand years of religious history and a deliberate philosophical departure.

One language gave birth to another. The data trail from *dharma* to *dhamma* is the history of a civilization deciding what to keep, what to simplify, and what to argue against.

---

*Next: using modern LLMs to actually read the Mahabharata — and asking hard questions about what the translation models understand vs. what they're guessing.*
