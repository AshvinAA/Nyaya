"""
============================================================================
 Nyaya - Ingestion Pipeline :: which base provision an overlay chunk amends
============================================================================
 WHAT THIS FILE IS ABOUT
 -----------------------
 An amending Act numbers its OWN sections: section 3 of the Labour
 (Amendment) Act, 2026 amends section 1 of the 2006 Act, its section 21
 inserts a new section 61A, and so on. Stage 7 of retrieval pairs a base
 chunk with the overlay chunk that changes it, so pairing on the overlay's
 own section_number is wrong. Every overlay chunk therefore carries an
 `amends_section` field naming the base provision(s) it changes.

 The value is read from the amending section's heading, which the 2026 Act
 writes in one fixed form:

     "২০০৬ সনের ৪২ নং আইনের ধারা ১৭ এর সংশোধন"            -> ["17"]
     "... ধারা ৩০৭ক ও ৩০৭খ এর সন্নিবেশ"                      -> ["307A", "307B"]
     "... ধারা ৩৪৫ক, ৩৪৫খ ও ৩৪৫গ এর সন্নিবেশ"               -> ["345A", "345B", "345C"]
     "... প্রথম তফসিল এর প্রতিস্থাপন"                          -> ["schedule_1"]
     "২০০৬ সনের ৪২ নং আইনের কতিপয় শব্দের সংশোধন"           -> ["*"]  (whole Act)
     "সংক্ষিপ্ত শিরোনাম ও প্রবর্তন" / "রহিতকরণ ও হেফাজত"   -> []     (amends nothing)

 Section ids use ASCII digits and the Latin letter the English text of the
 2006 Act uses for inserted sections (ক->A, খ->B, ...), so "৬১ক" matches
 the base text's "61A".

 Used by: metadata_tagger.py (tags overlay chunks), and the eval / QA
 scripts that check the mapping.
============================================================================
"""

import re

from ingestion.common import ascii_digits

# Bangla letter suffix of an inserted section -> the Latin letter the
# English text of the Act uses
_SUFFIX = {"ক": "A", "খ": "B", "গ": "C", "ঘ": "D", "ঙ": "E", "চ": "F"}
_ORDINAL_SCHEDULE = {"প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4,
                     "পঞ্চম": 5, "ষষ্ঠ": 6}

WHOLE_ACT = "*"

# heading must name the amended Act: "২০০৬ সনের ৪২ নং আইন(ের|ে)"
_TARGETS_2006_ACT = re.compile(r"২০০৬\s*সনের\s*৪২\s*নং\s*আইন")
_SECTION_LIST = re.compile(r"ধারা\s+((?:[০-৯]{1,3}[ক-চ]?)(?:\s*(?:,|ও|এবং)\s*[০-৯]{1,3}[ক-চ]?)*)")
_ONE_SECTION = re.compile(r"([০-৯]{1,3})([ক-চ]?)")
_SCHEDULE = re.compile(r"(" + "|".join(_ORDINAL_SCHEDULE) + r")\s*তফসিল")
_WORDS_EVERYWHERE = re.compile(r"কতিপয়\s*শব্দের\s*সংশোধন")


def normalize_section_id(bn):
    """'৬১ক' -> '61A', '১৭' -> '17'."""
    m = _ONE_SECTION.fullmatch(bn.strip())
    if not m:
        return None
    return str(int(ascii_digits(m.group(1)))) + _SUFFIX.get(m.group(2), "")


def amended_provisions(section_title):
    """Base-Act provisions an amending section changes, read from its
    heading. Returns a list of ids (see module docstring); [] when the
    heading does not amend a provision of the 2006 Act."""
    title = section_title or ""
    if not _TARGETS_2006_ACT.search(title):
        return []
    if _WORDS_EVERYWHERE.search(title):
        return [WHOLE_ACT]
    m = _SCHEDULE.search(title)
    if m:
        return ["schedule_%d" % _ORDINAL_SCHEDULE[m.group(1)]]
    m = _SECTION_LIST.search(title)
    if not m:
        return []
    ids = [normalize_section_id(num + suffix)
           for num, suffix in _ONE_SECTION.findall(m.group(1))]
    return [i for i in ids if i]


def amends_section_field(section_title):
    """Flat string for the Chroma metadata field (scalars only):
    '17', '307A;307B', '*', 'schedule_1', or None."""
    ids = amended_provisions(section_title)
    return ";".join(ids) if ids else None
