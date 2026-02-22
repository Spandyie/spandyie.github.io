---
layout: post
title: "Using Modern LLMs to Read the Mahabharata"
date: 2026-03-28
tags: [mahabharata, llm, translation, nlp, claude, bhagavad-gita, sanskrit]
description: "I fed the Bhagavad Gita to modern translation models and asked hard questions. Here's what MITRA can translate, where it fails, and what Claude adds when you treat the LLM as interpreter rather than translator."
---

I fed the Bhagavad Gita to a machine learning model and asked it hard questions. Here's what happened.

---

## Why the Gita

The Bhagavad Gita is 700 verses embedded in the Mahabharata, the episode on the eve of the Kurukshetra War where the warrior Arjuna refuses to fight and the god Krishna explains, at length, why he must. It's among the most translated texts in history — hundreds of English translations spanning 150 years, from Ganguli's 1883 rendering to contemporary editions by Bhikkhu Bodhi (Buddhist perspective), S. Radhakrishnan (academic), Prabhupada (Vaishnava devotional), and Barbara Stoler Miller (literary).

This diversity of translations is the key property for evaluation. When five careful, qualified translators produce meaningfully different English versions of the same Sanskrit verse, the differences tell you something about what's genuinely hard to translate — where the Sanskrit resists simple mapping to English.

Machine translation systems are optimized to produce English output. But the interesting question isn't whether the MT output is fluent English. It's whether it captures what's philosophically at stake.

---

## The MITRA Model

MITRA (arXiv 2601.06400, released 2025) is the current state-of-the-art for Sanskrit→English machine translation. Trained on 1.74 million parallel pairs assembled from SuttaCentral, the Digital Corpus of Sanskrit, and other sources, it substantially outperforms previous baselines on standard evaluation sets.

The model architecture is a fine-tuned encoder-decoder transformer. Sanskrit goes in (IAST transliteration or Devanāgarī), English comes out.

For evaluation, I ran MITRA against three reference translations on a set of key Gita verses, then computed semantic similarity using sentence embeddings to quantify how close the MT output was to each human translation.

```python
from sentence_transformers import SentenceTransformer, util

sim_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_translation(mt_output: str, references: dict) -> dict:
    """Compare MT output to human reference translations."""
    mt_emb = sim_model.encode(mt_output, convert_to_tensor=True)
    scores = {}
    for ref_name, ref_text in references.items():
        ref_emb = sim_model.encode(ref_text, convert_to_tensor=True)
        scores[ref_name] = util.cos_sim(mt_emb, ref_emb).item()
    return scores
```

---

## Test Case: Bhagavad Gita 2:47

BG 2:47 is the most famous verse in the Gita, possibly the most quoted Sanskrit verse in the world:

> *karmaṇy evādhikāras te mā phaleṣu kadācana*
> *mā karmaphalahetur bhūr mā te saṅgo stvakarmaṇi*

Word-for-word: "In action only your right; not ever in its fruits. Don't let fruit-of-action be your motive; don't be attached to inaction."

Here's how four translators render it:

**Ganguli (1883)**: "Let right deeds be thy motive, not the fruit which comes from them."

**Prabhupada (1972)**: "You have a right to perform your prescribed duty, but you are not entitled to the fruits of action. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty."

**Sargeant (literal, 1984)**: "Let right deeds be your motive, not the fruit which comes from them. And live in the action of labor, not for its rewards."

**Miller (literary, 1986)**: "Be intent on action, not on the fruits of action; avoid attraction to the fruits and attachment to inaction."

The divergences are not stylistic preferences. They're different philosophical readings. *Adhikāras* can mean "right," "authority," "qualification," or "domain." Prabhupada adds "prescribed duty" because he's reading through Vaishnava theology, where *karma* has a specific meaning tied to caste obligations. Ganguli, writing for a general British audience in 1883, strips the theological frame entirely.

The MT output handles the denotative content adequately — the basic claim is preserved. It struggles, as all literal translation does, with *adhikāras*: the word appears in English as "right" or "authority" without the philosophical freight the word carries in the context of the Gita's argument about action, agency, and consequence.

---

## Where MT Fails: Philosophical Passages

The Gita's hardest verses aren't the ones with unusual vocabulary. They're the ones where the grammar is simple but the meaning is load-bearing.

BG 2:20:
> *na jāyate mriyate vā kadācin nāyaṃ bhūtvā bhavitā vā na bhūyaḥ*
> *ajo nityaḥ śāśvato yaṃ purāṇo na hanyate hanyamāne śarīre*

"This [ātman] is never born and never dies. It has not come into being, does not, will not. Unborn, eternal, permanent, ancient — it is not slain when the body is slain."

The MT handles the denotative content. But *ajo* (unborn) and *nityaḥ* (eternal) and *śāśvato* (permanent) and *purāṇo* (ancient) are near-synonyms stacked deliberately — each word emphasizes a different aspect of the ātman's timelessness. A fluent English sentence that uses only one or two of these words loses the rhetorical accumulation. The Sanskrit is trying to convince through repetition-with-variation, not through a single precise claim.

This is the kind of thing semantic similarity scores don't fully capture, but a careful reader notices.

---

## Claude as Interpreter

The most useful thing I found was treating Claude not as a translator but as an interpreter — giving it both the Sanskrit verse and the range of human translations, then asking it to analyze the philosophical content.

```python
import anthropic

client = anthropic.Anthropic()

def analyze_verse(verse_sanskrit: str, translations: dict) -> str:
    translations_text = "\n".join(
        f"{name}: {text}" for name, text in translations.items()
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Analyze this Bhagavad Gita verse:

Sanskrit: {verse_sanskrit}

Human translations:
{translations_text}

Address these questions:
1. What is the precise philosophical claim being made?
2. Where do the translators diverge, and why does that divergence matter?
3. How does this compare to analogous ideas in Western philosophy?
4. What does a word-for-word rendering fail to capture?"""
        }]
    )
    return response.content[0].text
```

The output on BG 2:47 is genuinely illuminating. Claude identifies that *adhikāras* encodes a claim about *scope of agency* — not "you have rights" in a liberal-political sense, but "your proper domain is the action, not the consequence." This is close to Epictetus's distinction between *prohairesis* (what is up to us: our will, judgments, desires) and *apoprohairesis* (what is not up to us: outcomes, reputation, others' actions).

The Stoic parallel is real and documented — scholars like Pierre Hadot have explored it. But the Gita's framing is different: it's not just about psychological equanimity. It's embedded in a cosmological claim about the nature of the self and action in a universe governed by dharmic law. Claude navigates this distinction clearly, noting that the Stoic reading captures the psychological dimension while missing the metaphysical one.

This is what makes LLMs useful for this kind of analysis. Not translation — MT handles that adequately. Interpretation: the ability to hold multiple translations in context, identify where they diverge, and reason about why the divergence is philosophically meaningful.

---

## Honest Assessment

**What MT does well**: denotative accuracy on straightforward verses, handling standard theological vocabulary, producing fluent English output at scale. MITRA's 1.74M training pairs have given it coverage across the Sanskrit-English translation space that no previous model approached.

**Where MT falls short**: rhetorical structure, deliberate repetition-with-variation, context-dependent philosophical terms whose meaning shifts depending on whether you're reading Vedānta, Sāṃkhya, or Vaishnava theology. The model doesn't know it's in a Gita; it just produces the most likely English mapping of each token.

**Where Claude adds value**: precisely the interpretive layer that MT doesn't reach. Given the Sanskrit, multiple human translations, and a specific analytical question, Claude can navigate the philosophical terrain, compare traditions, and identify the fault lines in the translation choices. This is synthesis, not translation.

The best pipeline is: MITRA for base translation, semantic similarity for evaluation, Claude for interpretation and comparison. Neither replaces the others.

---

## What This Research Area Is

I started this series because I grew up with these texts as sound before I understood them as language. Sanskrit prayers at dawn, Pali chants occasionally woven in. Now I work in machine learning, and the question I kept returning to was: what do our best models actually understand when they process these texts?

The answer is: more than I expected, and less than the texts deserve.

The morphological analysis, the topic modeling, the cross-lingual comparison — all of it traces the outer shape of something that has engaged some of the finest human minds across three thousand years. The data finds real structure. The translation models produce useful output. The LLMs can reason about the philosophical content at a level that's genuinely helpful.

But the texts themselves — the Rigveda's fire hymns, the Buddha's discourses, the Gita's dialogue on action and liberation — those aren't primarily data. They're arguments about how to live. The NLP pipeline is a lens, not a replacement for reading.

My grandmother knew that. She wasn't interested in what topic model would find in the Hanuman Chalisa. She was interested in what it said.

I'm trying to hold both things at once.

---

*Code for all five phases of this series is available in the [sanskrit-nlp repository](https://github.com/Spandyie). Corpora: SuttaCentral, GRETIL, Itihasa (HuggingFace: rahular/itihasa), MITRA (arXiv 2601.06400).*
