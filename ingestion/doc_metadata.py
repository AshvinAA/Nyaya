"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 5 metadata registry (doc-level defaults)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Every ingested document gets a document-level metadata record describing
 WHO produced it, WHAT kind of authority it carries, and whether its text
 is the law in force today. Per-chunk metadata (metadata_tagger.py) is
 derived from these defaults and from each chunk's position in the
 document hierarchy.

 The fields are deliberately two different axes (INGESTION_PIPELINE.md
 Stage 0):
   legal_status  = AUTHORITY: is this provision settled law?
                   (ratified / pending_ratification / superseded / unverified)
   in_force      = APPLICABILITY: is this the text that governs TODAY?
 The two axes stay separate because an amendment can be operative before
 it is settled: the 2025 Labour (Amendment) Ordinance was in_force=True
 while legal_status=pending_ratification. It has since been replaced by
 the Bangladesh Labour (Amendment) Act, 2026 (Act No. 43 of 2026, assented
 and in force 10 April 2026), which is the amendment_overlay in this corpus
 and is legal_status=ratified. The overlay rule still applies - the English
 base text stops at the 2018 amendments, so overlay and base are surfaced
 together with labels - but the label must follow legal_status, never
 assume "pending".

 The registry is keyed by doc_slug (see common.doc_slug). The corpus is
 small and hand-audited: adding a document to docs/ means adding (or
 checking the fallback for) its row here.

 NOTE: summarize_table() lives here too - not because tables have anything
 to do with metadata, but because the summary text is what carries the
 table's document metadata into the embedding (Stage 7), and it keeps
 table_extractor.py focused on extraction mechanics.

 Used by: table_extractor.py (table metadata blocks), metadata_tagger.py
 (per-chunk tagging), run_pipeline.py (doc_metadata.json output).
============================================================================
"""

from ingestion.common import normalize

# ---------------------------------------------------------------------------
# Per-document authority/applicability defaults, keyed by doc_slug
# ---------------------------------------------------------------------------
# source_type        primary_law | secondary_legal_guide | international_oversight
#                    | policy_advocacy - critically separates the actual wage
#                    gazette from CPD's *proposed* (never adopted) figures
# authority_rank     official_gazette > official_translation >
#                    unofficial_translation > reputable_secondary >
#                    advocacy_report  (Stage 6 near-dup winner rule)
# text_layer         consolidated_base | amendment_overlay | historical_base
#                    | None
DOC_META = {
    "bangladesh_labour_act_2006_english_upto_2018": {
        "source_type": "primary_law", "source_act": "Bangladesh Labour Act, 2006",
        "effective_date": "2006-10-11", "legal_status": "ratified",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "official_english", "text_layer": "consolidated_base"},

    "bangladesh_labour_rules_2015_english_unofficial": {
        "source_type": "primary_law", "source_act": "Bangladesh Labour Rules, 2015",
        "effective_date": "2015-09-15", "legal_status": "ratified",
        "in_force": True, "authority_rank": "unofficial_translation",
        "translation_status": "unofficial", "text_layer": "consolidated_base"},

    # Identities below were re-verified by reading the PDFs (2026-11-12).
    # The filenames are misleading; the comments say what each file really is.

    "bangladesh_labour_amendment_act_2026_bangla_gazette": {
        # Act No. 43 of 2026, Bangladesh Gazette Extraordinary 10 Apr 2026,
        # pp. 15785-15831 (dpp.gov.bd). s.1(2): in force at once. Replaced the
        # 2025 Amendment Ordinance. Bangla original; the PDF has NO usable text
        # layer (CID fonts), so its text comes from the hand transcript in
        # docs/transcriptions/<slug>/ (see ingestion/transcripts.py). Overlay
        # chunks carry amends_section (ingestion/amendments.py).
        "source_type": "primary_law",
        "source_act": "Bangladesh Labour (Amendment) Act, 2026",
        "effective_date": "2026-04-10", "legal_status": "ratified",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "original_bangla", "text_layer": "amendment_overlay"},

    "rmg_minimum_wage_gazette_dec2023_bangla": {
        # Ministry of Labour notification No. 40.00.0000.016.32.021.13.30 dated
        # 18 Dec 2023, Gazette Extraordinary 20 Dec 2023, pp. 28201-28211
        # (Minimum Wage Board portal). The operative garments wage schedule:
        # Grade 4 total Tk 12,500, effective 1 Dec 2023. Bangla original; same
        # no-text-layer problem as the 2026 Act - read from the hand transcript.
        "source_type": "primary_law",
        "source_act": "Minimum Wages Notification for the Garments Sector, 2023",
        "effective_date": "2023-12-01", "legal_status": "ratified",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "original_bangla", "text_layer": None},

    "bangladeshgagette2025": {
        # NOT the 2025 Ordinance. A one-page English rendering of Ministry of
        # Labour notice No. 40.00.0000.016.99.009.24-207: garment-sector annual
        # increment raised to 9% (5% + 4%), the extra 4% effective 1 Dec 2024.
        # The file is an unofficial English version, not the gazette itself.
        "source_type": "primary_law",
        "source_act": "Garments Sector Annual Increment Notification, 2024",
        "effective_date": "2024-12-01", "legal_status": "ratified",
        "in_force": True, "authority_rank": "unofficial_translation",
        "translation_status": "unofficial", "text_layer": None},

    "bangladeshgagetteinbanglishjul2021": {
        # Authentic English Text of the Bangladesh EPZ Labour Act, 2019
        # (Act No. II of 2019), S.R.O. No. 257-Law/2021, Gazette 27 Jul 2021.
        # A SEPARATE statute that applies only to EPZ workers - its section
        # numbers do not correspond to the Labour Act, 2006. text_layer=None
        # keeps it out of base/overlay pairing with the 2006 Act.
        "source_type": "primary_law",
        "source_act": "Bangladesh EPZ Labour Act, 2019",
        "effective_date": "2019", "legal_status": "ratified",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "official_english", "text_layer": None},

    "bangladeshgagettesep2015": {
        # Authentic English Text of the Labour Act, 2006 as it stood on
        # 28 Sep 2015 (Gazette of that date) - i.e. BEFORE the 2018 amendments.
        # Superseded as a whole by the "upto 2018" text, so it is hidden by the
        # default filter and only surfaces for historical queries. Its own
        # text_layer stops ingestion dedup from merging it with the current
        # base (a superseded twin must never demote the in-force chunk).
        "source_type": "primary_law",
        "source_act": "Bangladesh Labour Act, 2006",
        "effective_date": "2015-09-28", "legal_status": "superseded",
        "in_force": False, "authority_rank": "official_gazette",
        "translation_status": "official_english", "text_layer": "historical_base"},

    "legal_500_bangladesh": {
        "source_type": "secondary_legal_guide", "source_act": None,
        "effective_date": None, "legal_status": None,
        "in_force": False, "authority_rank": "reputable_secondary",
        "translation_status": "original_english", "text_layer": None},

    "report_of_the_committee_on_freedom_of_association": {
        "source_type": "international_oversight", "source_act": None,
        "effective_date": None, "legal_status": None,
        "in_force": False, "authority_rank": "reputable_secondary",
        "translation_status": "original_english", "text_layer": None},

    "revision_of_the_minimum_wage_of_rmg_workers_in_2023": {   # CPD study
        "source_type": "policy_advocacy", "source_act": None,
        "effective_date": "2024-03", "legal_status": None,
        "in_force": False, "authority_rank": "advocacy_report",
        "translation_status": "original_english", "text_layer": None},
}

# Safety net for a docs/ file with no registry row: conservative defaults
# (never in force, unknown everything) so an unmapped document can never
# masquerade as settled primary law.
DEFAULT_DOC_META = {"source_type": None, "source_act": None,
                    "effective_date": None, "legal_status": None,
                    "in_force": False, "authority_rank": "reputable_secondary",
                    "translation_status": "unknown", "text_layer": None}


def doc_metadata(slug):
    """Return a fresh per-document metadata dict (copied per chunk so no
    caller can mutate the registry itself)."""
    meta = dict(DEFAULT_DOC_META)
    meta.update(DOC_META.get(slug, {}))
    return meta


# ---------------------------------------------------------------------------
# Layer note: the Stage 0 overlay rule, in one function
# ---------------------------------------------------------------------------
def layer_note(meta):
    """Human-readable statement of this document's Stage 0 layer status -
    stored on chunks so the generation stage can surface the labels instead
    of silently picking a version (the 'both are surfaced' rule)."""
    layer = meta.get("text_layer")
    status = meta.get("legal_status")
    if layer == "amendment_overlay":
        if status == "pending_ratification":
            return ("Amendment overlay: operative text, ratification pending - "
                    "supersedes the consolidated base where they overlap")
        if status == "ratified":
            return ("Amendment overlay: enacted amendment not yet merged into "
                    "the consolidated base - supersedes the base where they "
                    "overlap")
        return "Amendment overlay layer (status: %s)" % status
    if layer == "consolidated_base":
        return ("Consolidated base text (Act + ratified amendments)"
                if status == "ratified"
                else "Consolidated base text layer (status: %s)" % status)
    if layer == "historical_base":
        return ("Historical text of the Act - superseded, shown only to say "
                "what the law used to be")
    return "Not part of the consolidated law layers"


# ---------------------------------------------------------------------------
# Table summary (embed-side text for Stage 2b table chunks)
# ---------------------------------------------------------------------------
def summarize_table(rows):
    """Generate the natural-language summary that GETS EMBEDDED for semantic
    search (the plan: summary embeds, JSON generates). Deliberately rule-based
    so the demo has zero API dependency."""
    if not rows:
        return "Empty table."
    header = rows[0]                               # first row = column labels
    cells = [c for row in rows for c in row if c]  # every non-empty cell
    width = len(header)                            # number of columns
    return ("Structured data table with %d columns: %s. "
            "It contains %d data rows and %d populated cells; sample values: %s.") % (
        width,
        ", ".join(header[:6]) or "unlabeled",
        max(0, len(rows) - 1),
        len(cells),
        ", ".join(cells[:5]) or "none")
