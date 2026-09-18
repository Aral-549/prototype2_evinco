# PAIMANA — front end

Next.js 16 + Tailwind v4 client for the PAIMANA early-warning API.

## Run

The API must be running first (from the repo root):

```bash
DEBUG=false PORT=8012 .venv/bin/python run.py
```

Then:

```bash
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8012 npm run dev     # http://localhost:3000
```

Production:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8012 npm run build
NEXT_PUBLIC_API_BASE=http://localhost:8012 npx next start -p 3100
```

`NEXT_PUBLIC_API_BASE` defaults to `http://localhost:8012`.

---

## Why it looks like this

The previous dashboard put four KPI tiles, a ten-field simulator, the
assessment, the forecast curve, six statutory flag cards, five SHAP rows and
the directives **on one screen at once**. Everything competed; nothing led.

This rebuild applies the three Apple HIG principles literally.

**Clarity — one question per view.** Hierarchy comes from type size and
negative space, not from boxes. The overview asks only *how much public
capital is exposed, and which projects account for it*: one hero figure, three
quieter supporting stats, one ranked list.

**Deference — the chrome recedes.** Hairline separators instead of card
borders, a near-white ground instead of pure white, no gradients or glows.
Colour is **semantic only** — red/orange/yellow/green always mean risk tier and
are never used decoratively, so a red number is always information.

**Depth — progressive disclosure.** Detail is one tap away rather than
permanently on screen:

| Layer | Answers | Contains |
|---|---|---|
| Overview (`/`) | What do I look at today? | Hero exposure, 3 stats, tier rail, ranked list |
| Project sheet | Why this project? | Score + what set it, slip forecast, flags, drivers, actions |
| Evidence (`/evidence`) | Why should I believe it? | Benchmarks, out-of-time, cohort analysis, lead time |

Density reductions that came out of this, measured against the old page:

- Leaderboard: **9 columns → 4** (identity, risk, exposure, a way in)
- Statutory flags: **6 cards always → triggered only** (usually 1), with
  "Show all six checks"
- Drivers: **5 always → top 3**, with "Show all"
- Risk mix: **4 count tiles → one proportional rail**
- Simulator: **removed from the overview entirely**

## Notes

- **No webfont is loaded.** `system-ui` resolves to SF Pro on Apple hardware
  and a native face elsewhere, so typography can never fail to load and take
  the layout with it. (The old dashboard sourced its entire layout from a CDN
  and rendered as unstyled text without internet.)
- **Indian numbering throughout.** `₹1.5 lakh Cr`, `₹9,536 Cr` — `en-IN`
  grouping, not `en-US`.
- **Tabular numerals** on every figure so values don't jitter as they update.
- **Deep links.** `/?project=MOSPI_701415` opens that project's sheet, so a
  reviewer can be sent straight to the project under discussion.
- **Honest empty states.** When the backend reports a figure has not been
  measured (HTTP 503), the UI says so rather than rendering a zero.
- Dark mode is supported via `prefers-color-scheme`; light is the default.
- Respects `prefers-reduced-motion`.
