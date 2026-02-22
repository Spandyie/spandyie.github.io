---
layout: post
title: "Why Sanskrit Nearly Broke NLP (And How Researchers Fixed It)"
date: 2026-02-28
tags: [sanskrit, nlp, morphology, cltk, linguistics, technical]
description: "Modern NLP assumes tokenization is trivial. Sanskrit laughs at this assumption. A walkthrough of sandhi, compound decomposition, and the tools researchers built to handle it."
---

Modern NLP pipelines begin with tokenization. Split the text into words. It's a preprocessing step, almost an afterthought — something you do before the real work starts. BERT's tokenizer runs in milliseconds. GPT's BPE tokenizer handles virtually any text you throw at it.

Sanskrit made early NLP researchers stop dead before they got to step two.

---

## The Tokenization Problem

Most languages separate words with whitespace. Sanskrit does too, loosely — but the written words often don't correspond to the underlying morphological units, because of a process called **sandhi** (संधि, *joining*).

Sandhi is phonological fusion at word boundaries. When two words are spoken or written in sequence, the final sound of the first word and the initial sound of the second word interact according to a set of rules, sometimes merging into a single sound, sometimes transforming each other. The written text reflects this fusion.

A few examples:

| Word 1 | Word 2 | Fused form | Rule |
|--------|--------|------------|------|
| *rāma* | *uvāca* | *rāmovāca* | a + u → o |
| *tat* | *ca* | *tacca* | t + c → cc (assimilation) |
| *devī* | *api* | *devyapi* | ī + a → ya (glide insertion) |
| *sat* | *ānandam* | *sadānandam* | t + ā → dā (voicing) |
| *manas* | *īśa* | *manasīśa* | s + ī → sī (visarga sandhi) |

The fused form is what appears in the written text. A tokenizer splitting on whitespace gets *rāmovāca* as a single token. It has no idea this is two words. And without knowing it's two words, it cannot look either up in a dictionary, parse either morphologically, or understand what the sentence means.

There are dozens of sandhi rules. They interact. They're partly deterministic (given the sounds, the output is predictable), but splitting — going from *rāmovāca* back to *rāma* + *uvāca* — is hard because it requires knowing which splits are morphologically valid. The same fused string could sometimes result from multiple underlying word pairs.

---

## The Compound Problem

Sanskrit compounds (*samāsa*, समास) make the tokenization problem worse. Sanskrit freely concatenates stems into compound words, and these compounds can be long.

The compound types matter for interpretation:

- **Tatpuruṣa**: determinative compounds — *rājakumāra* = *rāja* (king) + *kumāra* (son) = "king's son"
- **Bahuvrīhi**: exocentric compounds that describe something else — *kamalalochana* (lotus-eyed) doesn't refer to a lotus or an eye, but to a person with lotus-shaped eyes
- **Dvandva**: copulative compounds — *ahorātra* = *aha* (day) + *rātra* (night) = "day-and-night"

A single Sanskrit compound can carry what English would express as a full noun phrase. Epic texts routinely produce 8-10 word compounds. Decomposing them requires not just finding the morphological boundary points but choosing the semantically coherent parse — because the same character sequence might be segmentable in multiple ways, not all of which are real words.

---

## The Morphological Complexity

Even after you've correctly split sandhi and decomposed compounds, each token is still a morphologically complex unit. Sanskrit nouns decline across:
- 8 **cases** (vibhakti): nominative, accusative, instrumental, dative, ablative, genitive, locative, vocative
- 3 **genders** (liṅga): masculine, feminine, neuter
- 3 **numbers** (vacana): singular, dual, plural

That's 72 possible inflected forms for a single noun stem — and many of them look different. A root like *deva* (god) appears as *devaḥ*, *devam*, *devena*, *devāya*, *devāt*, *devasya*, *deve*, *deva* depending on case, plus another 16 forms for dual and plural.

Verbs are similarly complex: 10 tenses, multiple moods, 3 persons, 3 numbers, active and middle voices.

And because the grammar is encoded in endings rather than word order, Sanskrit has essentially free word order. You cannot use positional heuristics. Parsing requires full morphological analysis at every token.

---

## The Tools That Emerged

The computational linguistics response to Sanskrit's complexity has been substantial. Here's the landscape:

**Pre-deep-learning**: The Sanskrit Heritage Platform (Gérard Huet, INRIA) built a rule-based morphological analyzer covering the full declension and conjugation system. Oliver Hellwig's work produced sandhi-aware segmenters trained on gold-standard annotated data. These rule-based systems have very high precision but require extensive hand-crafted linguistic knowledge to maintain and extend.

**CLTK — Classical Language Toolkit** is the Python-accessible entry point for most researchers now. It wraps multiple back-end analyzers and provides a unified API:

```python
from cltk import NLP

# Initialize the Sanskrit NLP pipeline
# Downloads models on first run (~500MB)
cltk_nlp = NLP(language="san")

# Analyze a verse from the Bhagavad Gita
verse = "karmaṇy evādhikāras te mā phaleṣu kadācana"
doc = cltk_nlp.analyze(text=verse)

for token in doc.tokens:
    print(f"{token.string:<20} lemma={token.lemma:<15} pos={token.pos}")
```

Expected output:
```
karmaṇi             lemma=karman         pos=NOUN
eva                 lemma=eva            pos=PART
adhikāras           lemma=adhikāra       pos=NOUN
te                  lemma=yuṣmad         pos=PRON
mā                  lemma=mā             pos=PART
phaleṣu             lemma=phala          pos=NOUN
kadācana            lemma=kadācana       pos=ADV
```

CLTK covers tokenization, lemmatization, morphological tagging, and sandhi splitting. The quality is good for classical Sanskrit; it degrades on Vedic Sanskrit, which has different morphological patterns and more archaic vocabulary.

**IndicBERT** (`ai4bharat/indic-bert`) is a multilingual BERT model pretrained on 12 Indic languages — including Sanskrit, Hindi, Bengali, Tamil, and others — using a combined dataset of ~9 billion tokens. For Sanskrit NLP tasks that involve understanding semantics (rather than morphological parsing), IndicBERT provides strong embeddings out of the box. It handles Devanāgarī script natively.

**ByT5-Sanskrit** (arXiv 2409.13920) takes a different approach entirely. Standard transformer tokenizers split text into subword units using BPE or SentencePiece — both of which fail badly on Sanskrit, because the relevant morphological units don't align with high-frequency byte sequences. ByT5-Sanskrit operates at the **byte level**, representing text as a sequence of raw UTF-8 bytes. This bypasses the tokenization problem entirely. The model learns to work with the raw character stream.

The results are significant: ByT5-Sanskrit substantially outperforms subword-tokenized baselines on Sanskrit NLP tasks, particularly for morphological analysis where the structure is in the characters rather than the words.

---

## The Pali Comparison

Pali simplified Sanskrit's morphological system deliberately. The canon was transmitted orally for centuries — by monks who had to memorize thousands of pages of text and reproduce them precisely. A language with fewer sandhi rules, simpler conjugations, and reduced compound depth is a language that's easier to transmit without error.

| Feature | Sanskrit | Pali |
|---------|----------|------|
| Cases | 8 | 6 (dropped some instrumental/locative distinctions) |
| Numbers | 3 (dual retained) | 2 (dual dropped) |
| Sandhi rules | 40+ | ~15 (simplified) |
| Verb tenses | 10 | 4 |
| Max compound depth | 10+ stems | 3-4 stems |

The sound changes from Sanskrit to Pali are systematic. *Dharma* becomes *dhamma* (consonant cluster simplified). *Karma* becomes *kamma* (same pattern). *Nirvāṇa* becomes *nibbāna* (the *rv* cluster becomes *bb* by assimilation). These aren't corruptions — they're regular, predictable transformations of a related language.

This makes Pali meaningfully easier to parse computationally, though still considerably harder than English. The tokenization problem is reduced. The morphological complexity is lower. A standard modern NLP pipeline can get further with Pali before breaking.

---

## Where the Field Stands Now

In 2025, Sanskrit NLP is a genuinely active research area. The MITRA model (arXiv 2601.06400) achieves state-of-the-art machine translation for Sanskrit→English and Pali→English, trained on 1.74M parallel pairs. ByT5-Sanskrit pushes the frontier on morphological analysis. IndicBERT provides strong multilingual embeddings.

The gap between Sanskrit NLP and English NLP is closing — not by making Sanskrit easier, but by building models robust enough to handle its complexity. Byte-level modeling, multilingual pretraining, and large parallel corpora are all doing work here.

The next post in this series applies these tools to the Vedic corpus: what does BERTopic find when you run topic modeling across the Rigveda, Atharvaveda, and Upanishads? What are the Vedas actually *about*?

That question has occupied scholars for three thousand years. The data has a view.
