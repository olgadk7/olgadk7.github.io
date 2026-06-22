---
title: "Introducing Bobbin: Turn Any Text Into A Concept Graph"
layout: post
blog: true
category: blog
author: Olga Kahn
permalink: /blog/introducing-bobbin
comments: true
---

# Introducing Bobbin: Turn Any Text Into A Concept Graph

We read a lot of dense text — lecture notes, long chats with an LLM, research articles — and most of it collapses into a wall we never open again. **[Bobbin](https://heybobbin.com)** is a small tool I built to fix that: paste in some text and it weaves the ideas into a navigable concept graph — each idea as its own entry, linked to the others it relates to.

<div style="margin:1.75em 0">
  <div style="border:1px solid #e4e2d8;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);background:#fbfaf6">
    <iframe src="https://heybobbin.com" title="Bobbin — live demo" loading="lazy" style="width:100%;height:620px;border:0;display:block"></iframe>
  </div>
  <p style="font-size:.85em;color:#9a9788;text-align:center;margin:.6em 0 0">A live, read-only demo — drag the graph, switch views. <a href="https://heybobbin.com">Open the full app ↗</a> to add your own text.</p>
</div>

## What it does

Drop in a transcript or some notes and Bobbin:

- **Extracts the concepts** and the relationships between them, deduping repeated ideas instead of listing them twice.
- **Types each node** for whatever the text is about — `person`, `work`, `concept`, `mood` — so it isn't locked to one domain.
- Lets you read it four ways: a **Digest** (each concept with its connections), a **Relationships** table, an interactive **Graph**, and an **Insights** view that surfaces the hubs and the loosely-connected gaps.

Keep adding text and it merges into the same graph. Sign in and your graphs save as projects. There's a live demo on the homepage — a music-theory graph — so you can poke around before adding your own.

## Under the hood

Bobbin is a Next.js app with Supabase for auth and storage (row-level security, so your graphs stay yours), and the extraction runs through Claude behind a server route. The name is a bobbin — the spool that thread winds around — which is roughly what it does with ideas.

## Try it

👉 **[heybobbin.com](https://heybobbin.com)** — paste something you've been meaning to make sense of.
