#!/usr/bin/env python3
"""Draft a monthly build log from what actually happened.

Reads git history across your project repos and new posts on this site,
then writes a markdown draft with the machine-knowable parts filled in
and the human parts left as prompts. You edit it, then send it.

    ./tools/buildlog.py                  # last 30 days, repos under ~/Dev
    ./tools/buildlog.py --since 45
    ./tools/buildlog.py --root ~/code --out drafts/october.md

Nothing here sends anything. It writes a file and prints the path.
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys

# Repo directory name -> how it should read in the log.
DISPLAY_NAMES = {
    "olgadk7.github.io": "the site",
}

SKIP_DIRS = {"node_modules", ".git", "vendor", "_site", "dist", "build"}


def run(args, cwd):
    try:
        out = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def find_repos(root):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return []
    repos = []
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or name in SKIP_DIRS:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(os.path.join(path, ".git")):
            repos.append(path)
    return repos


def commits_since(repo, since_iso, author=None):
    args = ["git", "log", "--no-merges", f"--since={since_iso}",
            "--pretty=format:%h\x1f%ad\x1f%s", "--date=short", "--all"]
    if author:
        args.insert(2, f"--author={author}")
    raw = run(args, repo)
    if not raw:
        return []
    out = []
    seen = set()
    for line in raw.split("\n"):
        parts = line.split("\x1f")
        if len(parts) != 3:
            continue
        sha, date, subject = parts
        # collapse repeated subjects (amends, rebases across branches)
        key = subject.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"sha": sha, "date": date, "subject": subject.strip()})
    return out


def posts_since(site_repo, since_date):
    posts_dir = os.path.join(site_repo, "_posts")
    if not os.path.isdir(posts_dir):
        return []
    found = []
    for fn in sorted(os.listdir(posts_dir)):
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-(.+)\.(md|markdown|html)$", fn)
        if not m:
            continue
        try:
            date = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if date < since_date:
            continue
        title, permalink = m.group(4), None
        try:
            with open(os.path.join(posts_dir, fn), encoding="utf-8") as fh:
                head = fh.read(2000)
            t = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', head, re.M)
            if t:
                title = t.group(1)
            p = re.search(r"^permalink:\s*(\S+)\s*$", head, re.M)
            if p:
                permalink = p.group(1)
        except OSError:
            pass
        found.append({"date": date, "title": title, "permalink": permalink})
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", type=int, default=30, help="days back (default 30)")
    ap.add_argument("--root", default="~/Dev", help="where your repos live")
    ap.add_argument("--author", default=None, help="filter commits by author")
    ap.add_argument("--max-commits", type=int, default=8, help="per repo, in the draft")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    today = dt.date.today()
    since_date = today - dt.timedelta(days=args.since)
    since_iso = since_date.isoformat()

    repos = find_repos(args.root)
    if not repos:
        sys.exit(f"No git repos found under {args.root}. Try --root.")

    site_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    lines = [
        f"# Build log — {today.strftime('%B %Y')}",
        "",
        f"*Draft covering {since_iso} to {today.isoformat()}. "
        "Edit before sending — the machine only knows what it can see.*",
        "",
        "## What I shipped",
        "",
    ]

    active = 0
    for repo in repos:
        commits = commits_since(repo, since_iso, args.author)
        if not commits:
            continue
        active += 1
        name = os.path.basename(repo)
        label = DISPLAY_NAMES.get(name, name)
        lines.append(f"**{label}** — {len(commits)} commit"
                     f"{'s' if len(commits) != 1 else ''}")
        for c in commits[: args.max_commits]:
            lines.append(f"- {c['subject']}")
        if len(commits) > args.max_commits:
            lines.append(f"- …and {len(commits) - args.max_commits} more")
        lines.append("")

    if not active:
        lines += ["Nothing committed in this window. Say that plainly, "
                  "or widen it with --since.", ""]

    lines += [
        "## What I learned",
        "",
        "<!-- The thing you now know that you didn't a month ago. One specific",
        "     thing beats three vague ones. Include what broke. -->",
        "",
        "## What I'm stuck on",
        "",
        "<!-- A real open question. This is the section people reply to,",
        "     so make it answerable. -->",
        "",
        "## What I wrote",
        "",
    ]

    posts = posts_since(site_repo, since_date)
    if posts:
        for p in posts:
            url = f"https://olgakahn.com{p['permalink']}" if p["permalink"] else ""
            suffix = f" — {url}" if url else ""
            lines.append(f"- {p['title']}{suffix}")
    else:
        lines.append("<!-- No new posts in this window. -->")
    lines += ["", "---", "", "*Reply to this — it comes straight to me.*", ""]

    out_path = args.out or os.path.join(
        site_repo, "_drafts", f"buildlog-{today.strftime('%Y-%m')}.md"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"Draft written to {out_path}")
    print(f"{active} repo(s) with activity, {len(posts)} post(s) in the window.")


if __name__ == "__main__":
    main()
