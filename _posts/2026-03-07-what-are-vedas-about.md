---
layout: post
title: "What Are the Vedas Actually About? A Data Perspective"
date: 2026-03-07
tags: [vedas, topic-modeling, nlp, sanskrit, humanities]
description: "The Vedas are 3,500 years old and scholars still debate what they mean. I asked a topic model. Here's what the data says — and what my grandmother would think of the answer."
---

The Vedas are 3,500 years old, and scholars still argue about what they mean.

Not because the texts are unclear — though they often are — but because the question "what are the Vedas about" turns out to be multiple different questions depending on who's asking, when, and with what stake in the answer. Brahmin ritualists, Vedānta philosophers, colonial-era Orientalists, Indian nationalists, and twentieth-century comparative mythologists have all read the Vedas and found different things at the center.

I asked a topic model. The answer is interesting.

---

## What the Vedas Are

Quick orientation, for anyone who didn't grow up with Sanskrit prayers:

The Vedas are four collections of ancient texts: the **Rigveda**, **Sāmaveda**, **Yajurveda**, and **Atharvaveda**. The Rigveda is the oldest and most studied — roughly 1,028 hymns composed between 1500 and 1200 BCE, addressed to deities and structured for ritual recitation. The Sāmaveda is mostly drawn from the Rigveda, set to melody for use in soma sacrifices. The Yajurveda contains ritual formulas. The Atharvaveda is different in character — more folk, more applied, full of spells, healing charms, and agricultural prayers.

Attached to each Veda are supplementary texts: Brāhmaṇas (ritual instructions), Āraṇyakas (forest texts, for meditation), and Upanishads (philosophical dialogues). The Upanishads were composed later, roughly 800-200 BCE, and represent a significant intellectual shift: from ritual invocation to metaphysical inquiry.

This arc — from hymn to philosophy, from fire sacrifice to pure thought — is what the data should, if it's working, be able to detect.

---

## What Topic Modeling Does

Topic modeling, stripped of jargon, is pattern-finding in large text collections. You feed in thousands of documents. The algorithm notices which words appear together more often than chance would predict. Those clusters of co-occurring words are called *topics*. Documents are then characterized by which topics they're "about," in what proportions.

The modern approach (BERTopic) improves on the older one (LDA) by using transformer embeddings: each document is represented as a point in semantic space rather than a bag of word counts. Documents that *mean* similar things cluster together, even if they use somewhat different vocabulary. For texts where word choice is theologically precise — and Vedic Sanskrit is nothing if not precise — this matters.

---

## What the Data Finds

The corpus: Rigveda, Atharvaveda, and a representative set of Upanishads, sourced from GRETIL, processed through CLTK for Sanskrit normalization, and run through BERTopic with IndicBERT embeddings.

**The Rigveda**: The topic structure is dominated by four overlapping clusters.

*Agni* (fire) is the largest. Agni is the sacred fire, the intermediary between humans and the gods, the carrier of offerings. Almost 200 of the Rigveda's 1,028 hymns are addressed to Agni. The topic cluster includes: fire, flame, sacrifice, offering (*havya*), priest (*hotṛ*), morning (*uṣas*). The fire isn't metaphorical — it's the literal ritual fire that received the soma oblations.

*Indra* is the second great cluster. Indra is the storm god, the dragon-slayer (*Vṛtrahan*, slayer of Vṛtra), the drinker of soma. The cluster includes: thunderbolt (*vajra*), rain, victory, battle, cattle, abundance. Indra is everywhere in the Rigveda because he embodies a core Vedic value: the power that breaks obstacles and makes things flow.

*Ṛta* — cosmic order, truth — threads through both. This is the concept that the universe operates according to an underlying law, and that ritual action maintains that law. The word *ṛta* appears throughout the Rigveda but becomes particularly dense in hymns to Varuṇa, the keeper of cosmic law. It's the ancestor of the concept of *dharma*.

*Ushas* (dawn) forms a smaller but distinct cluster: dawn, light, chariot, goddesses. The Ushas hymns are among the most beautiful in the Rigveda — lyrical in a way that most Vedic poetry isn't. They personify the dawn as a woman lighting the sky.

**The Atharvaveda**: The topic structure shifts notably. The cosmological concerns of the Rigveda — Agni, Indra, Varuṇa, Ṛta — are present but less dominant. What rises instead:

*Healing and disease*: herbs (*oṣadhi*), illness, healing, pain, body. The Atharvaveda is part medical text — it contains detailed treatments for diseases, attributed to various divine sources.

*Agriculture and protection*: cattle, field, harvest, abundance, home. Practical prayers for the kinds of things people actually needed: good crops, healthy livestock, protection from enemies.

*Spells and counters*: binding, releasing, enemy, protection, strength. The "magical" material that made earlier Western scholars uncomfortable.

The Atharvaveda wasn't used in the major śrauta (elite) rituals. It was a household religion, a village religion — the Vedic tradition in its folk expression. The topic model recovers this distinction without being told to look for it.

**The Upanishads**: The shift here is striking. The dominant clusters have almost nothing in common with the Rigveda.

*Ātman/Brahman*: self, consciousness, infinite, light, bliss. The great equation of the Upanishads is *ātman = brahman* — the individual self is identical with the ultimate reality of the universe. The Bṛhadāraṇyaka and Chāndogya Upanishads develop this in lengthy dialogues. The topic model identifies it as the most coherent cluster in the Upanishadic corpus.

*Mokṣa* (liberation): release, bondage, ignorance (*avidyā*), knowledge (*jñāna*). The goal is now freedom from the cycle of rebirth — a concept barely present in the Rigveda, which was more concerned with this-worldly abundance.

*Meditation and withdrawal*: breath (*prāṇa*), mind (*manas*), stillness, inner. The Upanishads introduce systematic contemplative practice as a path to knowledge.

---

## The Drift

This is what I find most striking: the vocabulary doesn't just change between Rigveda and Upanishads. The *concerns* change. The questions change.

Rigveda: "Please, Agni, carry our offerings to the gods, and bring us cattle and sons."

Upanishads: "What is the nature of consciousness? What survives death? What is the relationship between the self and the universe?"

A topic model doesn't know these are different questions. It just finds that the words that cluster together in one corpus are different from the words that cluster together in another. But the semantic distance is measurable, and it's large.

Scholars have called this the *Axial Age* shift — the period around 800-200 BCE when multiple civilizations independently developed inward, philosophical orientations. The Upanishads represent the Indian contribution to that shift. The data traces its outline.

---

## What My Grandmother Would Think

My grandmother recited these texts as prayers. She knew the Hanuman Chalisa, which draws on the Rāmāyaṇa tradition, and she knew some of the Upanishadic mahāvākyas — the great sayings: *tat tvam asi* (that thou art), *aham brahmāsmi* (I am Brahman).

She wasn't doing philosophy when she recited them. She was doing something older: maintaining a connection across time, transmitting syllables that had been transmitted across three thousand years of Indian civilization.

The data confirms she was transmitting something that evolved over that time — that shifted from fire ritual to pure metaphysics over a millennium. The texts she carried were the endpoint of a long transformation.

I find that worth knowing, even if she wouldn't have wanted a topic model to tell her.

---

*Next post: the same approach applied to the Pali Canon — what does BERTopic find in 1.7 million words of Buddhist teaching?*
