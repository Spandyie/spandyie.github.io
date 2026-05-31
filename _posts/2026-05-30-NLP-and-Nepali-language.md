---
layout: post
title: "NLP and the Nepali Language"
date: 2026-04-17
tags: [nepali, nlp, morphology, indic, low-resource]
description: "Why Nepali is harder for computers than its 17 million speakers suggest, and where the field stands today."
---

Modern spoken Nepali has largely collapsed the hraswa–dirgha distinction — the difference between short vowels (ह्रस्व: इ, उ) and their long counterparts (दीर्घ: ई, ऊ) that Sanskrit maintained carefully. The three sibilants Nepali inherited from Sanskrit (स, श, ष) — what teachers used to call the "patalo" and "moto" s-sounds — have similarly merged in most speakers' ears into effectively one or two sounds. Native speakers confuse बर्षि and बीर्गन्ज because the ikar and iikar sound identical; ष and श are used interchangeably in informal writing. The orthography carries complexity the ear no longer tracks.

This is the opposite of the Sanskrit NLP problem. Sanskrit's challenge is that sandhi *merges* words on the page that need to be split — the text is ambiguous. Nepali's problem is that the *writing system is more complex than the speech it represents*. A model sees three distinct characters (स/श/ष) and two vowel lengths (ि/ी) that are largely interchangeable in practice, and tries to learn signal from what is mostly noise.

To make the collapse concrete, here are minimal pairs that were once phonemically distinct and are now functionally homophones for most speakers:

| Short form | Long form | Distinct in Sanskrit? | Distinct in spoken Nepali? |
|------------|-----------|----------------------|----------------------------|
| दिन (*din*, day) | दीन (*dīn*, poor/wretched) | Yes | No |
| सुख (*sukh*, happiness) | सूख (*sūkh*, dryness) | Yes | No |
| शिव (*śiva*, Shiva) | शीश (*śīṣa*, head/top) | Yes | No |
| बुद्धि (*buddhi*, intellect) | वृद्धि (*vṛddhi*, increase) | Yes | Partially |

A reader sees the difference. A speaker rarely produces it. A model trained on web text has to decide whether the variation it observes between ि and ी is a meaningful morphological signal or just inconsistent orthography. Almost always, it's the latter.

---

## From Khas-Kura to Nepali

The language now called Nepali did not begin in Kathmandu. Its ancestor was **Khas-Kura** — the language of the Khas people, an Indo-Aryan group whose kingdom was centered on the **Sinja Valley of the Karnali region**, in what is today Jumla district in far western Nepal. The oldest surviving written record of the language is an inscription from the reign of Aśoka Challa (1255–1278 CE), a Khasa ruler whose kingdom at its peak stretched from Tibet to the Gangetic plain. An even earlier proto-Nepali inscription survives at Dullu in Dailekh District, dated to 1094 CE.

The Khas were not strangers to the Sanskrit textual world. The Mahabharata refers to them as *Khaśa* (खश): in the Sabhaparvan they appear among the peoples bringing tribute to Yudhishthira; in the Karna Parva they are placed among the hill peoples of the northwestern frontier. They sat at the edge of the Sanskrit-speaking world — their language carried Indo-Aryan inheritance, but they had no Pāṇinian grammarian standardizing how it should be written.

The language spread eastward through the Gorkha unification. When Prithvi Narayan Shah conquered the Kathmandu Valley in 1768 and established the modern Nepali state, the Gorkha court's language became the language of administration, the army, and eventually education across the new kingdom. It carried different names at different times: *Khas-Kura* (language of the Khas), then *Gorkhali* or *Gorkha Bhasha* (language of the Gorkhas), then *Parbatiya* (language of the hills). The name "Nepali" was officially adopted only in 1933, when the Gorkha Bhasa Prakashini Samiti renamed itself the Nepali Bhasa Prakashini Samiti.

The relevant point for NLP is this: when the Gorkha court adopted Devanagari as the script, they adopted it with its full Sanskrit orthographic conventions. The three sibilants. The hraswa–dirgha distinction. The complete classical vowel inventory including ऋ, ॠ, and the long ई/ऊ. But Khas-Kura had been a spoken Himalayan vernacular for centuries before that, developing its own phonology independently. The script brought distinctions the spoken language had never fully encoded. The gap between what written Nepali looks like and what spoken Nepali sounds like was not created by modern sloppiness. It was built into the language the moment a Sanskrit writing system was applied to a non-Sanskrit spoken form.

This is structurally different from how Hindi or Bengali developed. Both went through similar Prakrit→Apabhramsha stages, but their writing systems evolved alongside their phonologies. Nepali's writing system was effectively imported wholesale at a specific historical moment, carrying complexity its speakers never needed.

---

## Morphological Structure

Nepali is spoken by roughly 17 million people natively — in Nepal, across the Himalayan belt into Sikkim, Darjeeling, and Bhutan, and among a large diaspora. It is morphologically rich: verbs agree with subject in person, number, and gender, and nouns take postpositions rather than prepositions to mark grammatical role. Word order is SOV (subject-object-verb) and relatively fixed in declarative sentences, which is different from Sanskrit's near-free order.

A simple sentence: *रामले भात खायो* (*Rām-le bhāt khāyo*, "Ram ate rice").

- *राम* (Ram) is the subject
- *-ले* is the ergative case marker (Nepali uses split ergativity — agents of perfective transitive verbs take *-ले*)
- *भात* (bhāt) is the direct object, unmarked because it is inanimate
- *खायो* (khāyo) is the verb "ate", marked for third-person singular masculine past

That sentence already encodes person, number, gender, tense, aspect, and case. None of this is unusual for Indo-Aryan, but it's far from what a modern English-trained NLP pipeline expects.

**Gender** in Nepali has simplified relative to Sanskrit — two genders (masculine and feminine) rather than three — but gender still drives verb agreement and adjective forms throughout. *केटा हाँस्यो* (boy laughed) vs *केटी हाँसी* (girl laughed) shows the verb root itself changing form.

**Sandhi** exists in Nepali but is considerably less complex than Sanskrit's full system. Most sandhi in modern Nepali is internal to compound words (*राजकुमार* < *rāja* + *kumāra*) rather than crossing word boundaries on the page. The tokenization problem that dominates Sanskrit NLP is much milder here.

---

## The Register System

The feature that most surprises outsiders — and most breaks NLP systems — is the **register system**. Nepali verbs conjugate differently depending on the social relationship between speaker and subject. There are four levels:

| Register | Pronoun | Example: *to eat* (खानु) | Used for |
|----------|---------|--------------------------|----------|
| Intimate (घनिष्टार्थ) | तँ | खान्छस् | Close friends, younger people, children |
| Ordinary (सामान्य) | ऊ / तिमी | खान्छ / खान्छौ | Everyday peers, familiar adults |
| Respect (आदरार्थ) | तपाईं / उहाँ | खानुहुन्छ | Elders, teachers, respected strangers |
| Highest respect | हजुर | खाइबक्सिन्छ | Royalty, aristocrats, deities |

The highest register uses the auxiliary **बक्स्नु** (*baksanu*, "to grant/deign"), borrowed from Persian via Mughal court influence. It marks the action as something the exalted subject deigns to perform — a courtly form preserved in royal proclamations, religious contexts, and traditional theater.

To make the morphological gap concrete, here is the same question — "Did you eat?" — across all four levels:

```
Intimate:   तैंले खाइस्?           (taĩle khāis?)
Ordinary:   तिमीले खायौ?          (timīle khāyau?)
Respect:    तपाईंले खानुभयो?       (tapāī̃le khānubhayo?)
Highest:    हजुरले खाइबक्स्यो?     (hajurle khāibaksyo?)
```

The verb stem changes. The case marker on the pronoun varies. The auxiliary is different. These are not surface variants of a single underlying form — they are four parallel morphological systems that a Nepali speaker switches between based on social context.

For NLP, this matters in a concrete way. Newspaper text uses the ordinary or respect register almost exclusively. Religious and royal texts use the highest register. Social media and conversation use the intimate and ordinary. A model trained predominantly on one slice cannot generalize to the others — it has never seen the verb stems or auxiliaries that show up in the slices it missed.

---

## The Written-Spoken Gap

The hraswa/dirgha collapse is one instance of a larger problem: **the Nepali that appears in digital text is not the Nepali people speak**.

Newspapers, government documents, and formal writing use a standardized literary register that most speakers don't produce in daily life. Verb forms are formal, vocabulary is heavily Sanskritized (using *विद्यालय* for school where conversation uses *स्कूल*), orthography is careful. This is the Nepali that most NLP training data consists of.

Spoken Nepali — the language of homes, markets, and social media — differs in four concrete ways:

**Orthographic noise**: hraswa and dirgha are written inconsistently because they sound the same. The three sibilants (स/श/ष) appear interchangeably in informal writing. A model trained on this data sees variation where there is none — it cannot learn a distinction that the speakers themselves cannot hear.

**Register gap**: the four-level verb system means that the formal register of news text and the intimate register of conversation are nearly different morphological systems. A model that learns one doesn't generalize to the other.

**Code-switching**: Nepali speakers mix freely with Hindi, especially in the Tarai, and with English everywhere. A typical message might be: *"ramro cha, aaja meet garne?"* — Nepali structure, Devanagari script optional, mixed vocabulary. Standard Nepali NLP pipelines treat the English tokens as unknown or noise.

**Romanized Nepali**: A significant proportion of digital Nepali is typed in Roman script. *"kasto cha?"*, *"khana khayo?"*, *"k gardai hunuhunchha?"* — this text exists in enormous quantities on WhatsApp, Facebook, and SMS, and is almost entirely absent from training sets. Most multilingual models have never seen Devanagari-Roman transliteration as input.

---

## What the Field Has Built

Nepali NLP sits in an awkward position. With 17 million native speakers it is not a small language, but the research community has treated it as low-resource, concentrating effort on Hindi, Bengali, Tamil, and Telugu — Indic languages with larger digital footprints and stronger institutional backing.

**IndicBERT** (`ai4bharat/indic-bert`) covers Nepali alongside 11 other Indic languages, pretrained on approximately 9 billion tokens across the set ([arXiv:2005.00085](https://arxiv.org/abs/2005.00085)). It handles Devanagari natively and performs reasonably on classification and NER. The limitation is its pretraining data: the Nepali corpus is heavily skewed toward formal text. It reflects literary Nepali — the register you'd find in *Kantipur* or *Gorkhapatra*, not the language of conversation.

**muRIL** (Multilingual Representations for Indian Languages, Google 2021) extends the multilingual BERT approach to 17 Indian languages including Nepali, with substantially more pretraining data than IndicBERT ([arXiv:2103.10730](https://arxiv.org/abs/2103.10730)). Its cross-lingual transfer capabilities are stronger, which helps for zero-shot tasks when annotated Nepali data doesn't exist.

**XLM-R** (Facebook, 2020) is the multilingual baseline researchers reach for when nothing Indic-specific is available. It covers 100 languages including Nepali, though Indic scripts weren't a priority during training.

**IndicTrans2** (AI4Bharat, 2023) is the current state of the art for English↔Nepali machine translation, trained on large parallel corpora across 22 Indic language pairs ([arXiv:2305.16307](https://arxiv.org/abs/2305.16307)). On the FLORES+ benchmark — Meta's evaluation suite covering 200 languages — Nepali scores meaningfully below Hindi and Bengali but substantially above many other South Asian languages. Inference looks like this:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
ip = IndicProcessor(inference=True)

src_lang, tgt_lang = "eng_Latn", "nep_Deva"
sentences = ["The mountain weather changes quickly."]

batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
inputs = tokenizer(batch, return_tensors="pt", padding="longest")
outputs = model.generate(**inputs, max_length=256, num_beams=5)
translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
final = ip.postprocess_batch(translations, lang=tgt_lang)
print(final[0])
# हिमालको मौसम छिटो परिवर्तन हुन्छ।
```

The output is grammatical, uses the ordinary register, and gets the postposition *-को* right. Pass in romanized input — *"kasto cha?"* — and the model produces garbage. Pass in a sentence in the intimate register and you'll get the ordinary register back, because that's what dominates the training data.

**Sangraha** (AI4Bharat, 2024) is the most carefully curated Indic pretraining corpus to date — deduplicated, quality-filtered, covering 22 languages including Nepali ([arXiv:2403.06350](https://arxiv.org/abs/2403.06350)). If you are building a Nepali-focused model from scratch, this is where you start.

---

## What's Missing

Compare this to Sanskrit. Sanskrit has a dedicated morphological analyzer (the Sanskrit Heritage Platform), a full NLP toolkit in CLTK, byte-level models adapted to its morphological complexity (ByT5-Sanskrit), and over 1.7 million parallel sentence pairs in MITRA alone. Nepali has none of that. No morphological analyzer that handles the register system. No corpus of spoken Nepali. No dedicated toolkit equivalent to CLTK.

The gaps are specific:

- **No morphological analyzer that handles register**: A tool that takes *खाइबक्स्यो* and returns "verb=खानु, register=highest, tense=past" does not exist. The multilingual models embed it somewhere in their weights, but you cannot query it.
- **No spoken-Nepali corpus**: There is no equivalent of the Switchboard or Buckeye corpora for English. The Nepali that gets transcribed and annotated is almost always read speech from news broadcasts.
- **No Romanized-Nepali model**: WhatsApp Nepali, the most-used digital register, has effectively zero NLP coverage.
- **No code-switching tooling**: Models cannot reliably handle Nepali-Hindi-English mixing, despite this being how a large fraction of urban Nepali is actually produced.

The most significant gap isn't modeling — multilingual transformers handle the surface forms well enough for many applications. The gap is **annotated data for tasks that require linguistic depth**: morphological analysis across registers, NER in code-switched text, speech systems for informal spoken Nepali.

There is a quiet irony here. Sanskrit attracted serious computational attention partly through cultural prestige — it is a language scholars have cared about for millennia, and that care generated funding, datasets, and research infrastructure. Nepali, spoken by millions of living people who need working tools right now, has attracted less. The field's priorities don't always align with where the practical need is highest.

The models that exist work reasonably on literary Nepali — the careful Devanagari of newspapers and government documents. The language people actually type on their phones — Romanized, code-switched, intimate-register — is still largely invisible to them. Until training data catches up to how Nepali is spoken and written in practice, the gap between what the orthography shows and what the ear hears will keep surfacing as model error.
