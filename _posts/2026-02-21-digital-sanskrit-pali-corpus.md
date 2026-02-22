---
layout: post
title: "The Digital Sanskrit/Pali Corpus: What Exists and Why It's Hard"
date: 2026-02-21
tags: [sanskrit, pali, nlp, linguistics, heritage]
description: "A survey of the major Sanskrit and Pali corpora — SuttaCentral, GRETIL, Itihasa, MITRA — and why ancient texts are genuinely hard to digitize and parse."
---

Every morning of my childhood, my grandmother would recite the *Hanuman Chalisa*. I absorbed the syllables without understanding them — the rhythm of Sanskrit before I understood words, the cadence of Pali before I knew what Pali was. Now I spend my days training machine learning models, and recently I found myself asking: what happens when you feed these texts to a computer?

The short answer: the computer struggles. The longer answer is the subject of this series.

---

## Why Digitization Is Harder Than It Sounds

When most people imagine digitizing an old text, they picture scanning a book and running OCR. For medieval English manuscripts, that picture is roughly accurate. For Sanskrit and Pali, it unravels almost immediately.

The first problem is script diversity. Sanskrit has been written in over a dozen scripts across its history: Devanāgarī (the modern standard), Śāradā (used in Kashmir), Grantha (South India), Siddham (transmitted to East Asia with Buddhist texts), Bengali, Malayalam, and more. A single philosophical text might survive in multiple manuscript traditions, each in a different script, each with its own scribal errors and regional variants.

The second problem is encoding inconsistency. Before Unicode's 1990s standardization, scholars had developed half a dozen competing romanization schemes for Sanskrit: IAST (International Alphabet of Sanskrit Transliteration), Harvard-Kyoto, SLP1, Velthuis, ISO 15919. A text digitized at one institution in the 1980s might be incompatible — not just different-looking, but structurally incompatible — with one digitized at another institution in 1995. Normalizing across these schemes is tedious work that requires domain expertise.

The third problem is that Sanskrit and Pali have features that break basic NLP assumptions. More on that below.

---

## What Now Exists

The good news is that the last decade has produced some exceptional resources. Here are the four I consider foundational for this research series.

**SuttaCentral** (suttacentral.net) is the most complete digital resource for the Pali Canon. The full Tipitaka — monastic rules, discourses, and philosophical texts — is available with parallel English translations, structured as JSON accessible through an API. Bhikkhu Sujato's modern translations are particularly good: readable without being loose. The organization is meticulous. If you want to do NLP on early Buddhism, start here.

**GRETIL** — the Göttingen Register of Electronic Texts in Indian Languages — is the authoritative archive for Sanskrit source texts. Vedas, Upanishads, epics, philosophical treatises: if it exists in Sanskrit, GRETIL likely has it in plain text. The University of Göttingen has been building this since the 1990s. It's not pretty — no API, no JSON, just raw text files in mixed encoding schemes — but the philological quality is high.

**Itihasa** (from Rahul Arora et al., 2021) is 93,000 Sanskrit-English parallel sentence pairs extracted from the Ramayana and Mahabharata. It's available on HuggingFace (`rahular/itihasa`) and instantly loadable. This is what you reach for when you want a manageable Sanskrit corpus with aligned English translations for training or evaluation.

**MITRA** (2025) is the most ambitious: 1.74 million Sanskrit/Pali-English translation pairs, assembled from SuttaCentral, the Digital Corpus of Sanskrit, and other sources. The accompanying paper (arXiv 2601.06400) also releases state-of-the-art machine translation models for Sanskrit and Pali. If Itihasa is the starter dataset, MITRA is the serious one.

---

## Why Sanskrit Is a Nightmare for NLP

Modern NLP pipelines are built on assumptions. The most basic: words are separated by spaces, sentences end with punctuation, and a word is more or less a stable unit of meaning. English mostly satisfies these. Sanskrit systematically violates them.

**Sandhi** is the first challenge. Sandhi (संधि, *joining*) is a set of phonological rules that fuse words at their boundaries. In spoken Sanskrit, words don't end and begin cleanly — they flow together according to how the final sound of one word and the initial sound of the next interact. The written language reflects this: what you'd transcribe as two words in English becomes a single fused string in Sanskrit.

The classic example: *rāma + uvāca* (Rama said) becomes *rāmovāca*. The final *a* of *rāma* and the initial *u* of *uvāca* merge to *o* by a regular vowel-fusion rule. A tokenizer that splits on spaces will now give you one token — *rāmovāca* — and have no idea what to do with it.

There are dozens of sandhi rules. They interact. They're partly predictable, partly context-dependent. Splitting them requires a morphological analyzer that understands what the possible underlying words could be and can work backwards from the fused form. This is a hard computational problem that Sanskrit scholars have worked on for decades.

**Samāsa** (compound words) is the second challenge. Sanskrit permits — and frequently uses — compound words built by chaining multiple stems together. A word like *sarvajñatā* (omniscience) is already a compound (*sarva* + *jña* + *tā*), but Sanskrit compounds can go much deeper. Epic poetry routinely produces compounds with five, eight, ten constituent stems. Decomposing them requires understanding all the possible morphological boundaries and choosing the semantically coherent parse.

**Morphological complexity** is the third. Sanskrit has 8 grammatical cases, 3 genders, 3 numbers, and the information is encoded in word endings rather than word order. A single noun root might have 24 surface forms (8 cases × 3 numbers). Because the grammar is in the endings, word order is almost completely free — which means you can't use positional heuristics to figure out what's a subject and what's an object. You need full morphological analysis.

---

## Pali Is Comparatively Manageable

Pali is a close relative of Sanskrit — probably a Prakrit (vernacular dialect) from the Magadha region where the Buddha taught. It simplified Sanskrit's morphological system considerably, which was partly deliberate: the canon was transmitted orally for centuries, and a simpler phonological system is easier to memorize and reproduce.

Where Sanskrit has 8 cases, Pali has 6. Where Sanskrit has 3 numbers (singular, dual, plural), Pali dropped the dual. The sandhi rules are reduced and more regular. Compound depth is limited. This makes Pali meaningfully easier for computational analysis, though still far harder than English.

The relationship between the two languages is itself a research question: what Sanskrit words survived into Pali intact, which changed phonologically but kept their meaning, and which shifted meaning as they moved from the Brahminic tradition into the Buddhist one? That's the subject of a later post in this series.

---

## What Comes Next

This post has been orientation. The next post gets into the tools — CLTK, IndicBERT, ByT5-Sanskrit — and what happens when you actually try to tokenize a Sanskrit verse. We'll look at sandhi in detail and run some morphological analysis on Rigveda text.

The goal of this series isn't just technical. I want to understand, empirically, what these texts are *about* — what they cluster around, how they changed over time, what concepts travelled from one tradition to another. Data can't replace the lifetime of scholarship that Vedic and Buddhist studies have produced. But it can ask different questions, at scale, and sometimes find things that close reading misses.

My grandmother didn't know about BERTopic. But I think she'd appreciate the question.
