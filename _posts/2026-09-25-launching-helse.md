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
Everything below is live as of 2026-09-27 (pet profiles, "Add anything": live 2026-09-27).
Not live, so not claimed: heart-risk targets (plan item 7). Wearables = WHOOP only. Pets get no
AI and no targets.
-->

# Introducing Helse: Stop Reacting to Your Health

A couple of winters ago, a week before Christmas, my partner's brother-in-law died of a heart attack at 35. It wasn't a bolt from the blue — for months he knew something was wrong, and the system kept telling him it was stress. [I wrote about it here.](/blog/health-ownership) The autopsy found a 90% blockage and a pancreatic tumor nobody had gone looking for. Every warning sign had been there, assessed one at a time, and waved through.

That's the thing about how we do health: it waits for a crisis. Your data lives in a dozen portals, your labs come back "normal" ([which doesn't mean what you think it means](/blog/optimal-ranges)), and nobody — no doctor, no app — is holding the whole thread and watching it move.

So I built the tool I wish my family had had. It's called **Helse**, and today it's in beta.

## What it does

**Helse is the early-warning view of your health: every result in one place, read in context, watched over time — with the evidence behind every flag.**

You hand Helse anything with health data in it — a lab PDF, a photo of a paper report, a body scan, screenshots from your clinic's portal, a message from your doctor — and it works out what it is, files every result on the day it was drawn, and shows you what it found before anything is saved. From there, the app is laid out the way you'd think about your own health: what's true about you, what your data says, and what to do about it.

### About you: what Helse knows

- **Your profile.** Your age, your sex and your life stage — trying to conceive, pregnant, postmenopausal — with the date it started, because the same number means different things at different points in life.
- **Your health goals: where you're heading.** Better egg quality, more energy, a healthier heart. Goals decide what Helse pays attention to: which markers it puts first, how it reads your results, what goes into your supplement plan. What they never do is move a target — a healthy range doesn't change because you want it to.
- **Your health concerns: what you're dealing with.** An IVF cycle, a thyroid problem, a stubborn injury. Each concern gets its own page: a timeline of every test, appointment, medication and note, and your clinic's reports imported straight from a screenshot — so when something is worth watching, it doesn't become a number you've forgotten by the time you're back in the parking lot.

### Your results: what your data says

- **Every marker, tracked over time,** with a plain-language summary of what's improving and what's slipping. For the common markers, Helse knows how much each one naturally wobbles from one test to the next, so noise doesn't get reported as news.
- **Each number read in context.** Your sex, your age when the blood was drawn, your life stage, where you were in an IVF cycle, whether you'd fasted. When a comparison wouldn't be fair — a glucose from a draw that may not have been fasted, an estradiol taken mid-stimulation — Helse says "not graded" and why, instead of guessing.
- **The evidence, and how strong it is.** Next to your lab's "normal," Helse shows a target wherever the research supports one, with its source one click away — and says so plainly where it doesn't. Findings that aren't targets, like a trial where a supplement didn't help, appear as research notes, labelled for what they are. And when your own doctor gives you a number, it outranks everything.
- **Patterns, not just numbers.** When several heart-and-metabolic markers drift the wrong way together — each one still "normal" on its own — Helse flags it. That's exactly the kind of pattern that got waved through in my family.
- **Your wearable.** Connect a WHOOP, and your recovery, heart-rate variability, resting heart rate and sleep sit right alongside your labs — this week against your own usual, because a wearable's numbers only mean something against your own baseline.

![A Helse marker page for HbA1c: in the lab's normal range, but above the target, with its source and a history line](/assets/images/posts/helse/marker-verdict.png)

*A marker page on a demo profile (the numbers are made up): normal by the lab's range, above the target the research supports, with the source one click away.*

### What to do: how to act on it

- **What your results mean.** A read of your latest results through the lens of your goals: your priorities, why each one matters for what you're working toward, and a concrete next step for each. It works from the same targets and sources as the rest of the app, so it can't make up a number.
- **A supplement protocol built for you.** Dosed, and scheduled across your day — morning, midday, evening, bedtime, with food or without — with the interactions between supplements flagged, and anything your doctor prescribed treated as fixed. Want something changed? Chat with it: ask a question, or ask for a change and see exactly what would move before you apply it.

All of it is written for a better conversation with your doctor, not instead of one.

### For everyone you look after

The multi-profile part has a sillier origin: I was trying to keep track of my **dogs'** vet visits, scattered across clinics, and realized it was the same problem one step sideways. So Helse keeps a profile for everyone you look after, not just you — and yes, that includes your dog. His vet visits and vaccines on one timeline, a symptom diary and a diet diary, what he's taking and when, and his bloodwork and weight over the years, read against the vet lab's own range. The same idea as for you: catch the slow drift early, while there's still time to do something about it.

You can share any profile with a partner or a family member (or the dog sitter), to view it or to help keep it up to date.

## A concern up close: fertility and IVF

Concerns are where Helse goes deepest, and mine is fertility, so here's what one looks like in practice.

Monitoring scans and retrievals group into IVF cycles, so you can compare one round with the next — follicles, eggs, embryos — side by side. Your clinic's screenshots import straight in, and your recurring records, like scans and embryology reports, become forms you add to in seconds. When your clinician gives you a number for a stage — an estradiol target during stimulation — Helse holds every reading to it, at the right stage. A fertility view gathers the markers that matter — ovarian reserve, cycle hormones, thyroid, vitamin D, iron, B12 and folate, blood sugar — and shows which ones you haven't had tested yet. And your wearable's recovery is averaged across each cycle, so you can see how your body handled each round.

## Every number has a source

Most health apps hand you "optimal" ranges with nothing behind them. Before Helse showed a single target, every one was checked against the research, at the source.

Of the 60 markers people most often give "optimal" ranges for, only 10 had a target that held up — and those are the ones Helse uses. Several popular numbers didn't hold up at all: there's no agreed target for fasting insulin, and the general goal for ApoB is under 100 — the lower numbers you'll see quoted are for people at high risk, or have nothing behind them. For the other 43 markers, Helse tells you the honest answer: nothing beyond the lab's normal range.

Some evidence isn't a target at all, and Helse shows you that too. Women with higher vitamin D have had better IVF results in observational studies — but when a [randomized trial](https://pubmed.ncbi.nlm.nih.gov/33894153/) gave women with low vitamin D a large dose before IVF, pregnancy rates didn't go up. On your vitamin D page in Helse, that finding sits right under your number, labelled for exactly what it is.

## What it isn't

None of this is about becoming a hypochondriac or replacing your doctor. It's the opposite of anxiety: the calm of actually knowing, of catching the slow drift before it becomes an event. Health managed the way an actuary manages risk — trends, not snapshots; probabilities, not panic. It won't hand you confident numbers with nothing behind them, and it won't pretend "normal" means "fine." What it gives you is a better conversation with your doctor, with the whole picture in hand.

**Helse is in beta now at [gethelse.com](https://gethelse.com)**, free while it's in beta. Start with one lab PDF and see what a few years of it looks like on a chart.

A couple of practical things. It's passwordless, so you sign in with a code sent to your email — gethelse.com is a new domain, so that code sometimes lands in spam. Check there, and mark it "not spam" so it doesn't happen twice. And since I'm asking you to hand over lab results: your data isn't sold and isn't used for advertising, which the [privacy policy](https://gethelse.com/privacy) says in plain language rather than in legalese.

If your own health data has ever felt like it was working against you, come kick the tires — and tell me what's missing, at [hello@gethelse.com](mailto:hello@gethelse.com).

*Stop reacting to your health. Start staying ahead of it.*

— Olga
