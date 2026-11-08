---
name: commit-message-conventions
description: Apply this project's commit message format when writing a git commit message. Use when asked to write a commit, generate a commit message, or when about to run `git commit`.
user-invocable: true
---

## Commit Message Format

```
<type>(<scope>): <one-line, big-picture summary of what changed>

<optional: ONE short why-sentence, OR 2-3 one-line bullets — not both>
```

Keep the Conventional Commits `type(scope):` prefix already used in this repo's history (`fix(room):`, `feat(ledger):`, `fix(config):`, ...) — infer `type` and `scope` the same way past commits did (`git log --oneline` for precedent).

**Default to subject-only.** Most commits need nothing after the first line — the diff already shows what changed, and if the subject makes the reason obvious, stop there. Body text is the exception, earned only when the *why* genuinely isn't inferable from the subject or the code.

---

### Subject line

The big picture, in prose — not a truncated diff summary.

**Wrong:** `fix(config): update settings.py, add try/except in database.py`
**Right:** `fix(config): make local dev survive Postgres being down`

Describe the change at the level someone skimming `git log --oneline` wants, not the files it touched.

---

### Body — optional, pick at most ONE shape

If the subject doesn't fully explain the *why*, add a body — but only one of these, never both:

- **One short sentence** — the common case. A single line stating why, not a paragraph.
- **2-3 one-line bullets** — only when there are genuinely separate decisions, each independently non-obvious. One line each, no wrapping. If a reason needs more than one line to justify, it's not bullet material — cut it down or it doesn't belong in the commit message.

Never stack a why-paragraph *and* bullets — that's the same reasoning said twice. Pick whichever shape fits the number of decisions: one reason → one sentence; several independent reasons → bullets.

This applies even when the paragraph is framed as "context" and the bullets as "decisions" — a lead-in sentence setting up the problem, followed by bullets on what was decided, is still two shapes stacked. If the why splits into a general point plus per-decision reasons, fold the general point into bullet #1 instead of pulling it into its own paragraph above the list.

Target ~90 characters per line, hard stop at 100. If a sentence or bullet doesn't fit, it's carrying too much — cut a clause, don't wrap to a second line. A bullet that needs a second line to finish its reasoning is a sign it's actually two reasons, or one reason explained twice.

**Wrong** (restates the diff, not a reason):
```
- Added RoomConfigUpdated event
- Updated RoomConfig aggregate
- Added test for config update
```

**Wrong** (right idea, way too long — each bullet is a paragraph):
```
- Used a dedicated event instead of folding into RoomCreated so lobby-only
  config edits don't replay the full creation payload on rebuild, which
  matters because replay cost scales with event count
```

**Right** (one line, no wrap):
```
- Dedicated event, not folded into RoomCreated — avoids replaying full payload
- Validation lives in the command, not the aggregate
```

---

### Never include

- A restated list of changed files/functions/classes — that's what `git diff` and `git log -p` are for.
- A bullet that states *what* changed without saying *why*.
- Both a why-sentence and bullets in the same commit.
- Bullets or sentences that wrap past one line — if it doesn't fit on one line, it's over-explained; cut it.

---

### Full example

The project's actual `bcdfacf` commit was too long — a 3-line why-paragraph *plus* three wrapped, multi-line bullets, saying the same thing twice. Tightened to this skill's rules (bullets only, since there are two independent decisions):

```
feat(room): track per-player entity setup and add a way to start a game

- entity_id per player, not one flag — partial setup survives a restart
- assign/start split into separate commands, each enforcing its own invariants
```

That's the ceiling. Most commits should land shorter — subject line only, or subject plus one short sentence.

A second real draft (`EntityOrchestrator`) made the "context sentence + bullets" mistake — a 4-line lead-in paragraph explaining *why sequencing is needed*, followed by two more-wrapped bullets on the specific decisions:

```
feat(entity): add EntityOrchestrator to bootstrap a personal entity with its ledger

- Entity + Ledger are separate event-sourced writes, no shared transaction — needs sequencing
- Service shape (one method per use case), since more composed flows are coming
- No starting balance — funding amount is a game-config concern, out of scope here
```

The sequencing point that was a standalone paragraph became bullet #1 instead — no separate context section above the list.

---

### Workflow

This skill only changes the *message template* — follow the standard commit workflow otherwise: `git status` / `git diff --staged` / `git log` for style precedent, draft the message in this format, stage the relevant files, commit via a heredoc (`git commit -m "$(cat <<'EOF' ... EOF)"`). All the usual git safety rules still apply: never `--amend` an existing commit unless asked, never `--no-verify`, never commit without being asked to.