# Page transcripts

Some official gazettes in `docs/` have no usable text layer. They are set in fonts with no Unicode map, so pdfplumber returns `(cid:NNN)` tokens, and Stage 1 routes every page to `visual_routing_required`. For those PDFs, each page was transcribed by hand from a rendered image of the page. `ingestion/transcripts.py` gives that text to the intake driver in place of the PDF's own text layer. In `routing_report.json` the page's label becomes `transcribed`, and the router's original verdict is kept as `pdf_label`. Every chunk built from a transcript carries `text_source: manual_transcription`.

| Folder | Source PDF | Pages |
|---|---|---|
| `bangladesh_labour_amendment_act_2026_bangla_gazette/` | `Bangladesh-Labour-Amendment-Act-2026_Bangla-Gazette.pdf`: Act No. 43 of 2026, Gazette Extraordinary 10 Apr 2026, pp. 15785–15831 | 47 |
| `rmg_minimum_wage_gazette_dec2023_bangla/` | `RMG-Minimum-Wage-Gazette-Dec2023_Bangla.pdf`: Ministry of Labour notification 40.00.0000.016.32.021.13.30 of 18 Dec 2023, Gazette Extraordinary 20 Dec 2023, pp. 28201–28211 | 11 |

Transcribed 2026-11-18 from page images rendered at 144 dpi. Not yet proofread by a second reader. The wage tables were checked arithmetically: in all 8 grade rows, basic + house rent + medical + transport + food equals the printed total (workers' Grade 4: 6,700 + 3,350 + 750 + 450 + 1,250 = Tk 12,500).

## Conventions

- One file per PDF page, `pNNN.txt` (1-based), UTF-8. The text is the original Bangla, transcribed verbatim, including its numerals and the printer's own errors.
- One paragraph per line. Visual line wraps are joined.
- `#` lines are page furniture (the masthead, running headers, page numbers, the price, the printer's imprint, signatures) or a transcriber's note. They are dropped before chunking and never reach the index. Two notes mark printing errors kept verbatim. 2026 Act s.73 says it substitutes "ধারা ২০৭" when it means section 307. s.78 jumps from clause (খ) to (ঘ).
- `>` lines are text the document *quotes*, i.e. the wording an amending Act inserts or substitutes. That text has its own numbering ("(ক)", "৬১ক।", "১।" in a schedule), and that numbering belongs to the 2006 Act, not to the amending Act. The chunker therefore treats `>` lines as body text of the amending section that quotes them.
- `|` lines are table rows. The first row of a run is the column header. The chunker makes one chunk per data row and prefixes it with the header. Wage rows that run across a page break are marked "(চলমান)" (continued). Employee rows are labelled "তফসিল-খ (কর্মচারী)" so that the employees' Grade 4 (Tk 12,800) can't be confused with the workers' Grade 4 (Tk 12,500).

## Adding a transcript

Create `docs/transcriptions/<doc_slug>/` where `<doc_slug>` is `ingestion.common.doc_slug(pdf_path)`, and add one `pNNN.txt` per page you transcribe. Any page that has a transcript uses it instead of the PDF text. Pages without one are routed as before.
