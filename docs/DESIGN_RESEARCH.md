# Design research and interface rationale

The Research Edition is **award-informed**, not an award claim and not a visual imitation of any single site. Its design translates serious scientific content into an interface that can be explored without hiding evidence boundaries.

## Reference patterns reviewed

| Reference | Pattern studied | Application in AMP Finder AI |
|---|---|---|
| [Webby Awards: Websites and Mobile Sites](https://winners.webbyawards.com/winners/websites-and-mobile-sites) | Strong data visualization makes complex information comprehensible and engaging | Interactive, labeled research charts with restrained visual hierarchy |
| [HHMI's Beautiful Biology, 2025 Webby science winner](https://winners.webbyawards.com/2025/websites-and-mobile-sites/general-desktop-mobile-sites/science/326158/hhmis-beautiful-biology) and [site](https://www.beautifulbiology.org/) | Visual-first discovery, collections, and an invitation to explore biology | Clear modules, progressive disclosure, and a visual sequence-to-evidence hero |
| [EBRAINS, 2024 Webby science nominee](https://winners.webbyawards.com/2024/websites-and-mobile-sites/general-desktop-mobile-sites/science/289723/ebrains) and [site](https://ebrains.eu/) | Mission-led framing with direct paths to data, tools, and services | Problem-first opening followed by Predictor, Data, Methods, and Library workspaces |
| [Awwwards data-visualization collection](https://www.awwwards.com/websites/data-visualization/) | Editorial data storytelling and interactive exploration | Dark research canvas, high-contrast signals, compact evidence cards, and Plotly exploration |
| [Farm Minerals case study](https://www.awwwards.com/farm-minerals-case-study.html) | Translate technical science into clear, credible, grounded storytelling | Scientific language, explicit provenance, restrained motion, and no speculative “AI discovers antibiotics” copy |

## Design principles

### 1. Start with the mission

The opening communicates the scientific problem and the exact value proposition before exposing controls. The phrase “Map peptide sequence to biological evidence” is deliberately narrower than “discover antibiotics.”

### 2. Use visual energy to direct attention

The obsidian, mint, and acid-lime system creates a contemporary biotechnology identity. Motion is limited to the decorative peptide orbit and respects `prefers-reduced-motion`.

### 3. Make evidence boundaries part of the interface

Toy-data warnings, model status, translational endpoint tables, sequence-domain checks, and interpretation guardrails are prominent—not hidden in a footer.

### 4. Reveal complexity progressively

The top-level information architecture is:

1. **Research overview** — problem, question, decision, and evidence ladder.
2. **Predictor Lab** — single and batch screening with audit exports.
3. **Data Observatory** — dataset health, biology, split profile, and provenance.
4. **Methods & validation** — protocol, model card, uncertainty, DOME, and reproduction.
5. **Scholar Library** — databases, primary methods, reporting standards, glossary, and critical-reading questions.

### 5. Prefer research tools over decorative dashboards

Each chart has a defined analytical purpose:

- length distributions expose sampling differences;
- charge–hydropathy plots connect biophysics to labels;
- composition-difference bars reveal potential shortcuts;
- score gauges show threshold and uncertainty boundaries;
- normalized feature positions compare a candidate with training distributions.

### 6. Keep the interface responsive and accessible

- layouts collapse at tablet and mobile widths;
- color is paired with labels, shapes, or status text;
- content uses semantic headings and real form labels;
- motion can be disabled by the operating-system preference;
- data remain available as tables and downloads, not charts alone.

## Content standard

Every public-facing scientific statement should be one of:

- externally sourced fact with a link;
- measured project result with dataset/model context;
- explicit assumption;
- planned method;
- limitation or unmeasured endpoint.

The interface must never turn a model score into a wet-lab result through typography, color, or wording.

