# src/corpus_parser.py

import json
import unicodedata
from pathlib import Path
# for 01
import xml.etree.ElementTree as ET
import re
from collections import OrderedDict
# end for 01
def load_json(fname):
    """Loads a JSON file with utf-8 encoding."""
    return json.loads(Path(fname).read_text(encoding="utf-8"))

def strip_accents(text: str) -> str:
    """Removes diacritics from a string."""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# for 01
def get_book_number_map():
    """Returns a dictionary mapping standard biblical abbreviations to their book numbers."""
    ot_map = {
        "GEN": 1, "EXO": 2, "LEV": 3, "NUM": 4, "DEU": 5, "JOS": 6, "JDG": 7,
        "RUT": 8, "1SA": 9, "2SA": 10, "1KI": 11, "2KI": 12, "1CH": 13, "2CH": 14,
        "EZR": 15, "NEH": 16, "EST": 17, "JOB": 18, "PSA": 19, "PRO": 20,
        "ECC": 21, "SNG": 22, "ISA": 23, "JER": 24, "LAM": 25, "EZK": 26,
        "DAN": 27, "HOS": 28, "JOL": 29, "AMO": 30, "OBA": 31, "JON": 32,
        "MIC": 33, "NAH": 34, "HAB": 35, "ZEP": 36, "HAG": 37, "ZEC": 38, "MAL": 39
    }
    nt_map = {
        "MT": 40, "MR": 41, "LU": 42, "JOH": 43, "AC": 44, "RO": 45,
        "1CO": 46, "2CO": 47, "GA": 48, "EPH": 49, "PHP": 50, "COL": 51,
        "1TH": 52, "2TH": 53, "1TI": 54, "2TI": 55, "TIT": 56, "PHM": 57,
        "HEB": 58, "JAS": 59, "1PE": 60, "2PE": 61, "1JO": 62, "2JO": 63,
        "3JO": 64, "JUDE": 65, "RE": 66
    }
    return {**ot_map, **nt_map}

def parse_single_citation_part(citation_part_str, default_book_abbr=None):
    """
    Parses a single part of a biblical citation string (e.g., "RE 1:1", "1:1", "RE 1:1-8", "2:1-3:22").
    Uses default_book_abbr if the book abbreviation is missing.
    Returns book_abbr, start_ch, start_v, end_ch, end_v.
    """
    citation_part_str = citation_part_str.replace('–', '-').strip()
    # Updated pattern to optionally match the book abbreviation
    pattern = re.compile(r"^\s*([1-3]?[A-Z]+)?\s*(\d+):(\d+)(?:-(\d+)(?::(\d+))?)?\s*$")
    match = pattern.match(citation_part_str)

    if not match:
        raise ValueError(f"Invalid citation format part: '{citation_part_str}'")

    book_abbr_match, start_ch_str, start_v_str, end_part1_str, end_part2_str = match.groups()
    book_abbr = book_abbr_match if book_abbr_match else default_book_abbr

    if book_abbr is None:
         raise ValueError(f"Book abbreviation missing and no default provided for citation part: '{citation_part_str}'")


    start_ch, start_v = int(start_ch_str), int(start_v_str)

    if end_part1_str is None:
        end_ch, end_v = start_ch, start_v
    elif end_part2_str is None:
        end_ch, end_v = start_ch, int(end_part1_str)
    else:
        end_ch, end_v = int(end_part1_str), int(end_part2_str)

    return book_abbr, start_ch, start_v, end_ch, end_v

def parse_citation(citation_str):
    """
    Parses a standard biblical citation string, potentially with multiple parts separated by ';'.
    Returns a list of tuples: [(book_abbr, start_ch, start_v, end_ch, end_v), ...]
    Handles missing book abbreviations in subsequent parts.
    """
    citation_str = citation_str.strip()
    parts = [part.strip() for part in citation_str.split(';')]
    parsed_ranges = []
    default_book_abbr = None

    for i, part in enumerate(parts):
        if not part:
            continue

        try:
            if i == 0:
                # For the first part, book abbreviation must be present
                book_abbr, start_ch, start_v, end_ch, end_v = parse_single_citation_part(part)
                default_book_abbr = book_abbr # Set default for subsequent parts
            else:
                # For subsequent parts, use the default book abbreviation if not specified
                 book_abbr, start_ch, start_v, end_ch, end_v = parse_single_citation_part(part, default_book_abbr)


            parsed_ranges.append((book_abbr, start_ch, start_v, end_ch, end_v))
        except ValueError as e:
            print(f"Warning: Could not parse citation part '{part}'. Skipping. Details: {e}")
            # Continue processing other parts even if one fails

    # Add debug print for parsed ranges after processing all parts
    # print(f"Debug: Final parsed ranges for citation '{citation_str}': {parsed_ranges}")

    return parsed_ranges

def load_and_parse_xml(xml_file_path):
    """
    Loads and parses the XML Bible file.
    Returns a dictionary structure, a book abbreviation map, and the translation name.
    """
    bible_data = OrderedDict()
    book_abbr_map = {}
    translation_name = "Unknown Translation" # Default value
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Extract translation name from the root element's 'translation' attribute
        translation_name = root.get('translation', translation_name)

        book_map = get_book_number_map()
        # Correctly map book number from XML to abbreviation from our map
        num_to_abbr_map = {v: k for k, v in book_map.items()}


        for book_elem in root.findall('.//book'):
            book_num = int(book_elem.get('number'))
            # Use the book number from the XML to get the abbreviation from our map
            if book_num in num_to_abbr_map:
                 book_abbr = num_to_abbr_map[book_num]
                 book_abbr_map[book_num] = book_abbr # Store mapping for later use

                 bible_data[book_num] = OrderedDict()
                 for chapter_elem in book_elem.findall('chapter'):
                     chap_num = int(chapter_elem.get('number'))
                     bible_data[book_num][chap_num] = OrderedDict()
                     for verse_elem in chapter_elem.findall('verse'):
                         v_num = int(verse_elem.get('number'))
                         bible_data[book_num][chap_num][v_num] = verse_elem.text.strip() if verse_elem.text else ""
            else:
                print(f"Warning: Book number {book_num} found in XML but not in the predefined book map. Skipping.")


    except FileNotFoundError:
        print(f"ERROR: XML file not found at {xml_file_path}")
        return None, None, translation_name # Return default translation name on error
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML file. Details: {e}")
        return None, None, translation_name # Return default translation name on error
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during XML parsing. Details: {e}")
        return None, None, translation_name # Return default translation name on error

    return bible_data, book_abbr_map, translation_name

#for 02
# In file: src/corpus_parser.py

import re
from functools import lru_cache

# SBLGNT specific verse counts. Note Revelation 12 has 17 verses.
VERSES_PER_CHAPTER_GREEK = {
    1: 20, 2: 29, 3: 22, 4: 11, 5: 14, 6: 17, 7: 17, 8: 13, 9: 21, 10: 11,
    11: 19, 12: 17, 13: 18, 14: 20, 15: 8, 16: 21, 17: 18, 18: 24, 19: 21, 20: 15,
    21: 27, 22: 21
}

@lru_cache(maxsize=1)
def get_verse_map(verse_counts: tuple) -> dict:
    """Creates a mapping from (chapter, verse) to a global verse index."""
    verse_map = {}
    index = 0
    sorted_counts = sorted(dict(verse_counts).items())
    for chapter, num_verses in sorted_counts:
        for verse in range(1, num_verses + 1):
            verse_map[(chapter, verse)] = index
            index += 1
    return verse_map

def parse_citation_to_indices(citation_str: str, verse_map: dict, verse_counts: dict) -> set:
    """
    Robustly parses a biblical citation string into a set of global verse indices.
    Handles multi-chapter, single-chapter, and single-verse formats.
    """
    indices = set()
    citation_str = citation_str.strip()

    # --- START OF CORRECTION ---
    # Handle special case for Aune's Rev 12:18 which is Rev 13:1 in SBLGNT
    if citation_str == "RE 12:18-13:18":
        citation_str = "RE 13:1-13:18" # Remap for parsing against SBLGNT map

    # Define patterns
    multi_chap_match = re.match(r'RE\s*(\d+):(\d+)-(\d+):(\d+)', citation_str)
    single_chap_match = re.match(r'RE\s*(\d+):(\d+)-(\d+)', citation_str)
    single_verse_match = re.match(r'RE\s*(\d+):(\d+)$', citation_str)

    # Use if/elif/else to handle different formats correctly
    if multi_chap_match:
        start_c, start_v, end_c, end_v = map(int, multi_chap_match.groups())
        for chap in range(start_c, end_c + 1):
            v_start = start_v if chap == start_c else 1
            v_end = end_v if chap == end_c else verse_counts.get(chap, 0)
            for verse in range(v_start, v_end + 1):
                if (chap, verse) in verse_map:
                    indices.add(verse_map[(chap, verse)])
    elif single_chap_match:
        chap, start_v, end_v = map(int, single_chap_match.groups())
        for verse in range(start_v, end_v + 1):
            if (chap, verse) in verse_map:
                indices.add(verse_map[(chap, verse)])
    elif single_verse_match:
        chap, verse = map(int, single_verse_match.groups())
        if (chap, verse) in verse_map:
            indices.add(verse_map[(chap, verse)])
    else:
        # If no pattern matches, raise an error to make debugging clear
        raise ValueError(f"Could not parse citation string: '{citation_str}'")
        
    return indices
#end for 02



