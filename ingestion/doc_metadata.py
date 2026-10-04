"""
============================================================================
 Nyaya - Ingestion Pipeline :: Stage 5 metadata registry (doc-level defaults)
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 Every ingested document gets a document-level metadata record describing
 WHO produced it, WHAT kind of authority it carries, and whether its text
 is the law in force today. Per-chunk metadata (stage5_metadata.py) is
 derived from these defaults and from each chunk's position in the
 document hierarchy.

 The fields are deliberately two different axes (INGESTION_PIPELINE.md
 Stage 0):
   legal_status  = AUTHORITY: is this provision settled law?
                   (ratified / pending_ratification / superseded / unverified)
   in_force      = APPLICABILITY: is this the text that governs TODAY?
 The 2025 Ordinance is the concrete case: in_force=True (operative from
 promulgation) while legal_status=pending_ratification - the system must be
 able to say "this is the current text AND its ratification is pending",
 never collapse the two into one flag.

 The registry is keyed by doc_slug (see common.doc_slug). The corpus is
 small and hand-audited: adding a document to docs/ means adding (or
 checking the fallback for) its row here.

 NOTE: summarize_table() lives here too - not because tables have anything
 to do with metadata, but because the summary text is what carries the
 table's document metadata into the embedding (Stage 7), and it keeps
 stage2b_tables.py focused on extraction mechanics.

 Used by: stage2b_tables.py (table metadata blocks), stage5_metadata.py
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
# text_layer         consolidated_base | amendment_overlay | None
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

    "bangladeshgagette2025": {                       # 2025 amendment ordinance
        "source_type": "primary_law",
        "source_act": "Bangladesh Labour (Amendment) Ordinance, 2025",
        "effective_date": None, "legal_status": "pending_ratification",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "original_bangla", "text_layer": "amendment_overlay"},

    "bangladeshgagetteinbanglishjul2021": {          # identity unconfirmed until
        "source_type": "primary_law", "source_act": None,  # manual transcription
        "effective_date": None, "legal_status": "unverified",
        "in_force": False, "authority_rank": "official_gazette",
        "translation_status": "original_bangla", "text_layer": None},

    "bangladeshgagettesep2015": {                    # Sept-2015 Bangla printing
        "source_type": "primary_law",                # of the Act (its p2 title
        "source_act": "Bangladesh Labour Act, 2006",  # block says so)
        "effective_date": "2006-10-11", "legal_status": "ratified",
        "in_force": True, "authority_rank": "official_gazette",
        "translation_status": "original_bangla", "text_layer": "consolidated_base"},

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
        return ("Amendment overlay: operative text, ratification pending - "
                "supersedes the consolidated base where they overlap"
                if status == "pending_ratification"
                else "Amendment overlay layer (status: %s)" % status)
    if layer == "consolidated_base":
        return ("Consolidated base text (Act + ratified amendments)"
                if status == "ratified"
                else "Consolidated base text layer (status: %s)" % status)
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
