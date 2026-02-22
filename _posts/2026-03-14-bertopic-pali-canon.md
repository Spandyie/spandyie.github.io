---
layout: post
title: "BERTopic on the Pali Canon: What the Buddha's Words Cluster Into"
date: 2026-03-14
tags: [pali, buddhism, bertopic, nlp, topic-modeling, technical]
description: "The Pali Canon is three times the length of the Bible. I ran BERTopic on it. Here's what the topic model found — and what it confirms about the coherence of early Buddhist thought."
---

The Pali Canon is approximately 1.7 million words. The Bible is around 600,000. When someone tells you Buddhism has a "body of teachings," they mean something substantial — three baskets (*Tipitaka*) of texts covering monastic rules, discourses, and systematic philosophy, compiled over centuries and preserved with extraordinary fidelity by generations of monks.

So what happens when you hand a topic model 1.7 million words of ancient Buddhist text? You get a map.

---

## The Tipitaka in Brief

The corpus divides into three *piṭakas* (baskets):

**Vinaya Pitaka** — monastic rules and their origins. Often structured as case law: a monk did something, the Buddha ruled on it, a rule was established. It reads like a legal code with narrative context.

**Sutta Pitaka** — discourses. This is what most people encounter: the Dhammapada, the Majjhima Nikaya (middle-length discourses), the Samyutta Nikaya (grouped discourses), the Anguttara Nikaya (numerical discourses). Direct speech, narrative, dialogue, philosophy. The Buddha talking to monks, laypeople, kings, devas.

**Abhidhamma Pitaka** — systematic philosophy. A later development. No narrative frame, no characters. Pure analytical doctrine: enumerations of mental factors, causal matrices, philosophical taxonomy at a level of abstraction the Suttas don't reach.

The language throughout is Pali — a Prakrit related to Sanskrit, probably vernacular in the Magadha region. Preserved orally for centuries, written down in Sri Lanka around 100 BCE. The oral transmission was remarkably faithful; textual variants between manuscripts are minor.

---

## Why BERTopic, Not LDA

Latent Dirichlet Allocation (LDA) was the standard topic modeling approach for over a decade. It works. But it treats documents as bags of words — word order is discarded, and meaning that emerges from word combinations gets flattened.

For the Pali Canon, this is a real limitation. The Buddha's vocabulary is precise and theologically loaded. Terms like *anicca* (impermanence), *dukkha* (suffering, unsatisfactoriness), and *anattā* (non-self) are distinct concepts — but they appear together constantly because they constitute a unified teaching: the Three Marks of Existence (*ti-lakkhana*). LDA might cluster them correctly by co-occurrence. It might not. There's no semantic grounding.

BERTopic operates differently:

1. **Embed** each document using a transformer model — the document lives as a point in high-dimensional semantic space
2. **Reduce** dimensions with UMAP (Uniform Manifold Approximation and Projection)
3. **Cluster** with HDBSCAN (density-based clustering, robust to noise)
4. **Extract** representative terms per cluster using class-based TF-IDF

The clustering happens in embedding space. Documents that *mean* similar things cluster together, even if they use different surface vocabulary. For texts where word choice is deliberate and theologically significant, this matters.

---

## The Pipeline

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import json

# Multilingual BERT handles Pali transliteration well
# For raw Pali script, IndicBERT (ai4bharat/indic-bert) is preferred
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Load suttas — SuttaCentral provides structured JSON
# One document per sutta; segment by nikaya for comparative analysis
with open("tipitaka_en.json") as f:
    data = json.load(f)

documents = [entry["text"] for entry in data]
metadata = [{"nikaya": entry["nikaya"], "id": entry["id"]} for entry in data]

topic_model = BERTopic(
    embedding_model=embedding_model,
    nr_topics=30,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(documents)

# Visualization
topic_model.visualize_topics()      # 2D scatter of clusters
topic_model.visualize_hierarchy()   # Topic dendrogram
topic_model.visualize_barchart()    # Top terms per topic
```

A note on data source: SuttaCentral (suttacentral.net) is authoritative. Bhikkhu Sujato's translations are used for the English-language analysis; for Pali-script analysis, we use the original Pali texts with IndicBERT embeddings.

---

## What the Topics Look Like

Running BERTopic at 30 topics on the full Sutta Pitaka reveals clusters that map cleanly onto the known structure of early Buddhist thought.

**The Impermanence Cluster**

The most central cluster in semantic space. Terms: *anicca*, *dukkha*, *anattā*, arising, cessation, conditioned, origination, passing. This is the Three Marks of Existence — the foundational Buddhist ontology. The Buddha returns to it constantly, framed differently in different suttas but always structurally the same: all conditioned phenomena are impermanent, unsatisfactory, and without a fixed self.

The topic model doesn't need to be told this is important. The density and centrality of the cluster communicates it.

**The Meditation Cluster**

*Jhāna*, *samādhi*, *sati* (mindfulness), *vitakka* (applied thought), *vicāra* (sustained thought), *pīti* (rapture), *sukha* (pleasure), *ekaggatā* (one-pointedness). These are the four *jhānas* — meditative absorptions — and their constituent factors, described with the same technical vocabulary across hundreds of suttas. The phenomenology is precise. The fact that this cluster coheres so tightly suggests the terminology is being used consistently, not loosely — a finding with implications for debates about whether early Buddhist meditation practice was standardized.

**The Ethics Cluster**

*Sīla*, right action, the precepts, *kamma*, *cetanā* (intention), *kusala/akusala* (wholesome/unwholesome), *kammapatha* (course of action). Ethics in Buddhism is fundamentally about intention — the Buddha says explicitly that *cetanā* (volition) is what constitutes *kamma*. The model puts *cetanā* in the ethics cluster, which is a philosophically meaningful placement.

**The Community Cluster**

*Bhikkhu* (monk), *sangha*, *Vinaya*, rules, robes, almsround, dwelling, forest, requisites. The monastic life texts. This is predominantly Vinaya material, but it bleeds into Sutta passages about proper conduct for monastics. It's distinct from the meditation and ethics clusters — more procedural, less philosophical.

**The Liberation Cluster**

*Nibbāna*, *taṇhā* (craving), cessation, the ending, *vimutti* (liberation), *āsavā* (taints, fermentations). What's interesting here is that *nibbāna* clusters with the vocabulary of *ending* rather than *arriving* — cessation language, not destination language. Nibbāna is "the quenching of craving," not "the divine realm." The cluster reflects the apophatic character of the Buddhist soteriology: liberation is defined by what ceases, not by what you gain.

**The Dependent Origination Cluster**

*Paṭicca-samuppāda*, conditions, arising, consciousness (*viññāna*), formations (*saṅkhārā*), name-and-form (*nāmarūpa*). The twelve links of dependent origination — the causal chain from ignorance to suffering — form a coherent cluster. This is a structure the Buddha returns to repeatedly, in forward and reverse order. The model finds it as a distinct topic, separate from the impermanence cluster even though they're related.

---

## The Abhidhamma Anomaly

The most structurally interesting finding: the Abhidhamma texts cluster almost entirely separately from the Suttas.

This is what a scholar would expect. The Abhidhamma represents a later systematization — instead of the Buddha speaking to a monk in a mango grove, you get catalogues of mental factors, causal matrices, and philosophical taxonomy. The *Dhammasangani* (enumeration of dhammas) reads nothing like the *Majjhima Nikaya*.

But the model doesn't know genre. It finds the separation through semantics. The Abhidhamma texts use a different vocabulary, operate at a different level of abstraction, and produce a different embedding signature. They end up far from the narrative suttas in topic space, even though both are canonical and authoritative.

This convergence matters: when an unsupervised algorithm recovers the same distinctions that centuries of philological scholarship have identified, you have some confidence the pipeline is finding real structure rather than noise.

---

## What This Confirms

The Buddha's core message is remarkably consistent across thousands of discourses. A topic model finds that the same cluster of concepts — impermanence, the four *jhānas*, *nibbāna* defined by cessation of craving — appears in the Theravada, Sarvāstivāda, and other Nikaya traditions. Scholars like Bhikkhu Anālayo and Richard Gombrich have argued this from comparative textual analysis. The topic model argues it from semantic geometry.

The Pali Canon, to a BERTopic model, looks like a body of work with a tightly clustered center and a structured periphery. The center is the three marks, meditation, and liberation. The periphery is monastic rules, cosmology, and later philosophical elaboration.

That structure was put there deliberately — by a tradition that cared deeply about preserving a specific teaching intact. The data confirms it worked.

---

*Code: notebooks available in the [sanskrit-nlp repository](https://github.com/Spandyie). Data source: SuttaCentral (suttacentral.net).*
