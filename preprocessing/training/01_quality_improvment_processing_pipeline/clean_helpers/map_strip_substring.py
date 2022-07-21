import re
from typing import List


def build_substring_stripper(list_of_substrings: List[str]):
    bad_string_regex = re.compile("|".join([re.escape(elt) for elt in list_of_substrings]))
    def strip_substrings(batch):
        return {
            **batch,
            "text": [bad_string_regex.sub("", text) for text in batch["text"]]
        }
    return strip_substrings


en_wiktionary_stripper = build_substring_stripper([
    'This entry needs pronunciation information',
    'Please try to find a suitable image on Wikimedia Commons or upload one there yourself!This entry needs pronunciation information',
    'You may continue to edit this entry while the discussion proceeds, but please mention significant edits at the RFD discussion and ensure that the intention of votes already cast is not left unclear',
    'This entry is part of the phrasebook project, which presents criteria for inclusion based on utility, simplicity and commonality',
    'If you are a native speaker with a microphone, please record some and upload them',
    'If you are familiar with the IPA then please add some!',
    'Feel free to edit this entry as normal, but do not remove {{rfv}} until the request has been resolved',
    'This entry needs quotations to illustrate usage',
    'If you are familiar with the IPA then please add some!This entry needs audio files',
    'Please see that page for discussion and justifications',
    'If you are familiar with the IPA or enPR then please add some!A user has added this entry to requests for verification(+) If it cannot be verified that this term meets our attestation criteria, it will be deleted',
    'This entry needs a photograph or drawing for illustration',
    'A user has added this entry to requests for deletion(+)',
    'Do not remove the {{rfd}} until the debate has finished',
    'This entry needs audio files',
    'If you come across any interesting, durably archived quotes then please add them!This entry is part of the phrasebook project, which presents criteria for inclusion based on utility, simplicity and commonality',
    '(For audio required quickly, visit WT:APR)'
])