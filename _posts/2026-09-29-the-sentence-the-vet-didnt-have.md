---
title: "The Sentence the Vet Didn't Have"
layout: post
blog: true
category: blog
author: Olga Kahn
summary: "Health-data “interoperability” sounds like plumbing. It's one question: does the next clinician know what the last one learned — and what you know? A bad night at the animal ER, the standard that already answers it, and the one page Helse now makes."
# Link previews (LinkedIn, X, iMessage): the SEO tag reads these, not `summary`.
description: "Does the next clinician know what the last one learned — and what you know? A bad night at the animal ER, the standard that already answers it, and the one page Helse now makes."
permalink: /blog/interoperability
comments: true
subscribe_tag: helse
published: false
---

<!--
DRAFT — 2026-09-29. Before publishing, check:
[ ] Everything claimed is live (it was on 2026-09-29): the Allergies & reactions card on every
    profile, the reader picking alerts out of a document, the Summary for a clinician / for the
    vet page (print / save as PDF, leave things out, the lab's own ranges, no AI prose), the
    dated weight on the first line.
[ ] NOT live and NOT claimed: an IPS/FHIR file export — mentioned only as what comes next.
[x] The link to /blog/helse works: the launch post went live on 2026-09-29.
[x] A picture: the Demo dog's summary page, made-up records (HIDE='.print\:hidden' node
    scripts/screenshot.mjs http://localhost:3000/dashboard/summary <demo dog id> <out> 1126, in the
    Helse repo). No social card yet: link previews fall back to the site default.
[x] Olga's calls (2026-09-29): the title stays; Rocco's night stays as written; the vet stays
    blameless ("there was nowhere for him to have known it from").
Links to gethelse.com carry utm_campaign=interoperability; share the post with ?utm_source=<channel>.
-->

# The Sentence the Vet Didn't Have

*What a bad night at the animal ER taught me about health data — and the one page you should be able to hand any doctor.*

A few nights ago my husband and I took our dog to the emergency vet. Rocco is a twelve-year-old black lab. He was restless and breathing hard, and at that hour you don't wait and see. The vet did the sensible things: listened to his chest, sedated him for chest x-rays — which showed nothing alarming — wrote a prescription with the first dose due at seven that morning, a couple of hours after the sedation, and sent us home.

Then, for the next thirty hours, Rocco couldn't eat, couldn't drink, and could barely stand. He missed his first doses because he couldn't get anything down. It passed. But it didn't have to go that way, because there was a sentence that would have changed the dose, and it wasn't in the room:

*The last time he had a sedative at home, he could barely walk for a day. He's very sensitive to it.*

I knew that. The vet didn't. And that isn't his failure: there was nowhere for him to have known it from. Rocco's records live in three clinics and my head, and at two in the morning my head was busy.

## What "interoperability" means when it's your night

Interoperability is one of those words the healthcare industry uses about plumbing: whether one system can send a record to another. I'd never say it to a friend. But strip the plumbing away and it's one question:

**Does the next clinician know what the last one learned — and what you know?**

Most of the time the honest answer is no. Every emergency room and every walk-in clinic starts from the same four questions — any allergies? what are you taking? what conditions do you have? what happened? — and the answers come from a frightened person, from memory, or from a bag of pill bottles. It goes about as well as you'd expect. A systematic review of 22 studies found that [the medication history taken when someone is admitted to hospital is wrong for up to two-thirds of patients](https://doi.org/10.1503/cmaj.045311) — a drug left out, a drug that isn't actually being taken — and in the studies that judged it, between a tenth and more than half of those errors mattered clinically. That review is two decades old. Ask anyone who has been through an ER lately whether the questions have changed.

For animals it's starker. There is usually no shared record at all. The emergency vet knows exactly what the owner remembers to say.

Here's the part that got me. Rocco's history *was* in Helse. Two days after that visit I'd uploaded his discharge papers: the exam, the x-rays, the antibiotic and how long to give it, the medications he'd been on before. Everything a vet would want — except the sentence that mattered, because it had nowhere to live. Helse had a timeline, and a timeline scrolls. A bad reaction to a sedative, whenever it happened, isn't an event; it's a standing fact about him, the kind that should be the first line anyone reads.

## The standard already exists

The most encouraging thing I found is that the medical world has already agreed on what that first page should look like. It's called the [International Patient Summary](https://hl7.org/fhir/uv/ips/) — HL7 publishes the technical form, and it's an ISO standard (27269). It was designed for exactly this situation: the minimum a clinician needs about a person they've never met, for unplanned care. The standard's own words for itself are "minimal and non-exhaustive."

Its three required sections are allergies and intolerances, medications, and problems. Then immunizations, results, procedures. Then, optionally, alerts, vital signs, pregnancy — and something it calls the patient story: a section in your own words.

Nobody in an emergency room is going to read a data file off your phone. But look at that list again. It's the intake. It's the four questions. So a sheet of paper in that shape is the first interoperable document — one any clinician will accept, because it looks like the thing they already do — and a file in the standard's format is the same content in a second form, for the day the clinic's system can take it.

## So Helse makes the page

Every profile in Helse — a person's or a pet's — now has two new things.

**Allergies & reactions.** A place for what a clinician must know before treating you: an allergy, a bad reaction to a drug or a sedation or a treatment, or a plain caution — hard to find a vein, anxious at the clinic. What it was to, what happened, when, how bad, and who says so. They're standing facts, not timeline entries, so they never scroll away. And when you add a discharge summary or a visit note that states one, Helse picks it out and asks whether to keep it.

**A summary for your clinician.** One page, in the standard's shape: who you are, with your weight and the date it was measured; allergies and reactions; medications; problems; visits and procedures; immunizations; tests and imaging; your latest lab results, each with its own lab's reference range; the last three months of symptoms; who has provided your care. You can leave things out before you print — a dermatologist doesn't need your fertility history, and you decide. Then print it, or save it as a PDF and keep it on your phone for the night you need it. For Rocco it says "Summary for the vet," and his weight is on the first line, dated, because that's what the dose is worked out from.

![The Demo dog's summary for the vet: who he is with a dated weight, then a reaction to sedation first, then medications, visits and procedures, vaccinations, and lab results with the lab's own reference ranges](/assets/images/posts/helse/summary-for-the-vet.png)

*The Demo dog's page — made-up records, the real layout. The first thing under his name is the sentence.*

Two things it deliberately doesn't do. It doesn't grade your results against Helse's own targets — a clinician wants the lab's range, not an app's opinion. And it carries no AI-written prose. The point of the page is to be believed, and a stranger believes dates and sources.

Rocco's page has the sentence on it now. His next vet will read it before anything else, which is the whole idea.

## Try it before you need it

If you use Helse, make the page now, while nothing is wrong: open your profile and click "Summary for a clinician." Read it the way a stranger would. Fix what's missing — the reaction you never wrote down, the medication you stopped. Then take it to your next appointment, and tell me what your doctor said: that's what decides the next step, which is the standard's own file format, for the first clinic that can take it.

[Helse is in beta, and free while it is.](https://www.gethelse.com/?utm_source=olgakahn.com&utm_medium=blog&utm_campaign=interoperability) I introduced it [here](/blog/helse).

— Olga
