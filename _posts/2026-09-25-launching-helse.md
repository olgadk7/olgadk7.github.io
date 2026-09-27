---
title: "Introducing Helse: Stop Reacting to Your Health"
layout: post
blog: true
category: blog
author: Olga Kahn
summary: "The health tool I wish my family had had — every result in one place, read in context, watched over time, with the evidence behind every flag. Now in beta."
permalink: /blog/helse
comments: true
subscribe_tag: helse
published: false
---

<!--
BEFORE PUBLISHING — this post may only claim what is live. Tick each:
[x] Catalog targets switched on (plan item 1 in CLAUDE.md) — 2026-09-26, 10 markers.
[x] WHOOP configured in production — connected 2026-09-26, full history synced.
[x] A screenshot: a marker page with a verdict — demo profile, added 2026-09-26.
[x] The 2025 optimal-ranges post fixed first — rewritten and live 2026-09-26.
Update as later plan items ship: research signals (item 5), own-history flags (item 6),
heart-risk targets (item 7), drifting-together flags (item 8).
-->

# Introducing Helse: Stop Reacting to Your Health

A couple of winters ago, a week before Christmas, my partner's brother-in-law died of a heart attack at 35. It wasn't a bolt from the blue — for months he knew something was wrong, and the system kept telling him it was stress. [I wrote about it here.](/blog/health-ownership) The autopsy found a 90% blockage and a pancreatic tumor nobody had gone looking for. Every warning sign had been there, assessed one at a time, and waved through.

That's the thing about how we do health: it waits for a crisis. Your data lives in a dozen portals, your labs come back "normal" ([which doesn't mean what you think it means](/blog/optimal-ranges)), and nobody — no doctor, no app — is holding the whole thread and watching it move.

So I built the tool I wish my family had had. It's called **Helse**, and today it's in beta.

## What it does

**Helse is the early-warning view of your health: every result in one place, read in context, watched over time — with the evidence behind every flag.**

- **It watches the whole story.** Lab PDFs, clinic reports, wearables, supplements, meds — Helse reads them for you (a screenshot of a clinic portal becomes a structured record), puts every appointment and note on one timeline, and tracks where each marker is heading, with a plain-language read of what's improving and what's slipping. When something's worth watching, it becomes a tracked concern — not a number you forget by the time you're back in the parking lot.
- **It reads each number in context.** Your sex, your age at the time of the test, your life stage, where you were in an IVF cycle, whether you'd fasted. When a comparison wouldn't be fair — a glucose from a draw that may not have been fasted, an estradiol taken mid-stimulation — Helse says "not graded" and why, instead of guessing.
- **It shows the evidence, and how strong it is.** Where research supports a target beyond the lab's "normal," Helse shows it, with its source and how solid it is. Where it doesn't, Helse doesn't invent one. And when your own doctor gives you a number, it outranks everything.

![A Helse marker page for HbA1c: in the lab's normal range, but above the target, with its source and a history line](/assets/images/posts/helse/marker-verdict.png)

*A marker page on a demo profile (the numbers are made up): normal by the lab's range, above the target the research supports, with the source one click away.*

The multi-profile part has a sillier origin: I was trying to keep track of my **dogs'** vet visits, scattered across clinics, and realized it was the same problem one step sideways. So Helse keeps a profile for each person you look after, not just you.

## If you're trying to conceive or going through IVF

This is where Helse goes deepest. A fertility view gathers the markers that matter — ovarian reserve, cycle hormones, thyroid, vitamin D, iron, B12 and folate, blood sugar — and shows which ones you haven't had tested. Monitoring scans and retrievals group into cycles, so you can compare one round with the next, with wearable recovery averaged across each. Clinic results import straight from a screenshot. Your supplement protocol treats your doctor's prescriptions as fixed. And when your clinician gives you a number for a stage — an estradiol target during stimulation — Helse holds every reading to it, at the right stage.

## What I learned building it

A year ago I argued that "normal" isn't "optimal." Building Helse, I went back and checked every "optimal" number I could find against the research, at the source.

Of the 60 markers people most often give "optimal" ranges for, only 10 had a target that held up. Seven more had real evidence that isn't a target — a link to lower mortality in one study, say, or a number the medical societies disagree on. Several popular numbers didn't hold up at all: there's no agreed target for fasting insulin, and the general goal for ApoB is under 100 — the lower numbers you'll see quoted are for people at high risk, or have nothing behind them. For the other 43 markers, the honest answer is "nothing beyond the lab's normal range," and Helse doesn't pretend otherwise.

Some evidence isn't a target at all. Women with higher vitamin D have had better IVF results in observational studies — but when a [randomized trial](https://pubmed.ncbi.nlm.nih.gov/33894153/) gave women with low vitamin D a large dose before IVF, pregnancy rates didn't go up. A finding like that belongs in front of you, labelled for what it is, not turned into a confident number.

## What it isn't

None of this is about becoming a hypochondriac or replacing your doctor. It's the opposite of anxiety: the calm of actually knowing, of catching the slow drift before it becomes an event. Health managed the way an actuary manages risk — trends, not snapshots; probabilities, not panic. It won't hand you confident numbers with nothing behind them, and it won't pretend "normal" means "fine." What it gives you is a better conversation with your doctor, with the whole picture in hand.

**Helse is in beta now at [gethelse.com](https://gethelse.com)**, free while it's in beta. Start with one lab PDF and see what a few years of it looks like on a chart.

A couple of practical things. It's passwordless, so you sign in with a code sent to your email — gethelse.com is a new domain, so that code sometimes lands in spam. Check there, and mark it "not spam" so it doesn't happen twice. And since I'm asking you to hand over lab results: your data isn't sold and isn't used for advertising, which the [privacy policy](https://gethelse.com/privacy) says in plain language rather than in legalese.

If your own health data has ever felt like it was working against you, come kick the tires — and tell me what's missing, at [hello@gethelse.com](mailto:hello@gethelse.com).

*Stop reacting to your health. Start staying ahead of it.*

— Olga
