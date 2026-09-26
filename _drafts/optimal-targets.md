---
title: "I Re-checked My Own Optimal-Ranges Table. One Row Survived Intact."
layout: post
blog: true
category: blog
author: Olga Kahn
summary: "Last year I published five 'optimal' numbers. This year I tried to source each one to a standard I'd defend to a doctor — and built Helse to refuse any number that fails it."
permalink: /blog/optimal-targets
comments: true
subscribe_tag: helse
published: false
---

# I Re-checked My Own Optimal-Ranges Table. One Row Survived Intact.

A year ago [I published a table](/blog/optimal-ranges): five biomarkers, the lab's "normal" beside an "evidence-based optimal." It's the reason Helse exists.

Then I had to put those numbers inside a product that grades real people's blood, and "I read it somewhere credible" stopped being good enough. So I set a rule for what a number has to survive before Helse will show it as a target:

1. It comes from a clinical guideline or a peer-reviewed study that states the threshold outright — not a blog, not a book, not a summary of one.
2. The source is fetched and the sentence containing the number is quoted verbatim.
3. It's a *target* — a level tied to lower risk, or a treatment goal — not a reference interval, and not a diagnostic cut-off wearing an "optimal" label.
4. Then someone whose only job is to refute it re-reads the source and tries.

Here's what that did to my own table.

| | 2025 post | What survived the rule |
|---|---|---|
| **hs-CRP** | < 1.0 mg/L | < 1 mg/L — intact (AHA/CDC) |
| **ApoB** | ≤ 80 mg/dL | < 100. Eighty is the *high-risk* goal; I'd applied it to everyone |
| **TSH, trying to conceive** | ≤ 2.5 mIU/L | Survived only as an IVF treatment goal — and a second society disputes even that |
| **Ferritin, women** | 50–90, "better fertility outcomes" | 50–150. A real floor, a real ceiling, and nothing about fertility: the floor comes from a trial about fatigue |
| **Fasting insulin** | < 6 µIU/mL | Gone. No guideline sets a fasting-insulin goal, and the best outcome study found it isn't fasting insulin that predicts risk |

Across everything Helse tracks, twenty-five targets passed. Nine that looked right failed the refuter — one was a risk cut-off for a complication, mislabelled as a goal; one applied a treatment target meant for women on thyroid medication to every woman alive. Forty-four markers have no defensible optimal at all, so Helse shows none. The numbers you'll find on longevity blogs — ApoB under 60, CRP under 0.5, insulin between 2 and 5 — none of them is a guideline target.

The honest version of "normal isn't optimal" is smaller than the marketing version. It's also the only version I'll ship.

## Where the gap is real

It's not nothing. A woman's ferritin of 20 is "normal" — the lab's floor is 16. A peer-reviewed threshold puts the point where haemoglobin starts to fall at about 25, and a randomised trial found fatigue improves with iron below 50. Three sources say a normal number isn't a fine one. HbA1c: the lab flags 5.7; the target that survived is 5.5. ALT in women: the lab allows 29; the gastroenterology guideline says 25.

And where it isn't: vitamin D, hs-CRP — the lab already agrees. Helse says so, rather than inventing a tighter band to look clever.

## The three people I have to look in the eye

**You.** A target in Helse is matched to you — sex, age, and the stage you're in — and it tells you what it was matched on, so "for women 18–50" never masquerades as "for you." The number your own doctor gave you beats every catalog row, and it's labelled as theirs.

And when Helse can't judge a reading, it doesn't. My glucose is 85. Good? Only if I'd fasted — and the report doesn't say. So the row reads *not graded: this draw's fasting status isn't recorded; the target assumes fasting*, with a button to fix it. An estradiol drawn mid-stimulation is never judged against a baseline target, because that verdict would be wrong by construction. "Not graded" is the feature.

**Your doctor.** Every target carries its provenance: the source, its tier, and how strong the evidence is — "low" when it's one cohort, "contested" when two societies disagree, which is exactly what the TSH row says. Every reading is judged at its own date, in the state it was drawn in. And the AI that writes your interpretation is forbidden to state a threshold: it isn't handed the targets yet, and until it is, it may not invent one.

**The critic.** Twenty-five rows, each with a quote you can check, entered one at a time after a human reads the source. Nine rejections and forty-four "no target" markers, on the record. If you show me a source that beats mine, the row changes — that's what the provenance is for. If you think a row shouldn't exist, you're probably arguing with a guideline, and I'd like to watch.

## What this means in the app

Today: enter the target your clinician gave you, on any marker, and every past reading is graded against it — or told why it wasn't. The sourced catalog goes in row by row as I approve it. One more row waits on a sentence no machine could fetch from behind a journal's paywall; it goes in when a human has read it.

The table from last year stays up, with a note pointing here. Being wrong in public and fixing it in public is the whole point of writing things down.

*Educational, not medical advice. A target is the start of a conversation with your provider, not a verdict — and Helse will tell you which of its numbers are strong, which are thin, and which are argued over.*

<p style="margin:40px 0;text-align:center"><a href="https://gethelse.com" style="display:inline-block;padding:14px 28px;background:#6e002c;color:#fff;border-radius:4px;font-size:16px;text-decoration:none">Try Helse &rarr;</a></p>
