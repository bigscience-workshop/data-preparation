# -*- coding: utf-8 -*-
"""MST BigScience PII Code

Original colab that is a source of this file is located at
    https://colab.research.google.com/drive/1086H3-LGMz3gX0pGy9ECgr8KflosSKso

# License

Copyright 2022 Authors of this Notebook

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# What is this colab?

This colab detects the following kinds of PII for all languages in BigScience.
Languages assumed are ["ar", "as", "bn", "ca", "en", "es", "eu", "fr", "gu", "hi", "id", "ig", "mr", "ny", "pa", "pt", "sn", "st", "sw", "ur", "vi", "xh", "yo", "zh", "zu"]

## Highest Risk
### Simple spans of characters:
*   **IDs [general]:** This is anything that is a sequence of 6 or more digits, as is common in identifiers for people internationally (national IDs, tax IDs, passport numbers, etc.), credit card numbers, IBAN codes, etc.
*   **Key [general]**: This is anything that is a sequence of digits and letters in the same string, optionally with spaces.  Common for Credit Card and API, SSH, GPG keys. (Privacy group doesn't have a regex for this)
*   **Email address**, **User name**: Strings using @
*   **IP address**: Digits with periods in them
*   **Phone number**: At least 7 digits with spaces in them
*   **License plate**: (Privacy group doesn't have cross-lingual handling for this, MST group doesn't have a regex for this)

### More complex spans: (WORK IN PROGRESS)
* **Full Names**: Requires additional NER package
* **Address**


## Lower Risk: (We're not doing)
*   **URL**
*   **Time**: dateparser dependency
*   **Date**: dateparser dependency
*   **Age**

"""


#@title Define highest risk PII. TODO: License plate
# NUMBER removed last minute due to false positives. See https://huggingface.slack.com/archives/C0307KE5UNT/p1647011702716159
high_risk_tags = {'KEY', 'EMAIL', 'USER', 'IP_ADDRESS'} # , 'NUMBER', "ID"}

"""# Regexes"""

#@title Get the less sophisticated MST regexes for High Risk scenarios (baseline comparison). Not language-specific; all are general.
import sys
import regex
# These are ordered so that we can return upon a match; no need to search for a substring.
year_patterns = [
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-yyyy or yyyy/yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # mm-yyyy or mm/yyyy or the same but with yy
  regex.compile(r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])"), # yyyy-mm or yyyy/mm
]

# Patterns for high-risk character strings
id_pattern = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([A-Za-z]*(?:[\p{Pd}]*\p{Nd}){6,})(?:$|[\b\s@?,!;:\'\")(.\p{Han}])'
# https://regex101.com/r/JQkmh8/2
# key_pattern = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[\s\p{Pd}]?){4,})(?:$|[\b\s\p{Han}@?,!;:\'\"])'
# https://regex101.com/r/JQkmh8/5
key_pattern = r'(?:^|[\b\s@?,!:;\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[ \p{Pd}]?){3,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])'
ipv4_pattern = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
ip_pattern = r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + r"|".join([ipv4_pattern, ipv6_pattern]) + ")(?:$|[\s@,?!;:\'\"(.\p{Han}])"

# https://regex101.com/r/EpA5B7/1
email_pattern = r'''
    (?<= ^ | [\b\s@,?!;:)('".\p{Han}<] )
    (
      [^\b\s@?!;,:)('"<]+
      @
      [^\b\s@!?;,/]*
      [^\b\s@?!;,/:)('">.]
      \.
      \p{L} \w{1,}
    )
    (?= $ | [\b\s@,?!;:)('".\p{Han}>] )
'''

# https://regex101.com/r/mOqi1s/3
#user_pattern = r'(?:^|[\s@,?!;:\'\")(\p{Han}])(@[^\s@,?!;:\'\")(]{3,})'
user_pattern = r'''
  (?<= ^ | [)(\s@,?!;:'"\p{Han}] )
  (@
    [^)(\s@,?!;:'"]{3,}
  )
'''
# Examples from https://regexpattern.com/phone-number/
# https://regex101.com/r/lZZ0XP/4
# Also matches MLS numbers
# phone_pattern = r'(?:^|[\s\'\"(\p{Han}])((?:\+\p{Nd}+[ \/.\p{Pd}]*)?(?:(?:\(\+?\p{Nd}+\))?(?:[ \/.\p{Pd}]*\p{Nd})){7,}(?:[\t\f #]*\p{Nd}+)?)(?:$|[\s@,?!;:\'\"(.\p{Han}])'

id_regex = regex.compile(id_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
key_regex = regex.compile(key_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
ipv4_regex = regex.compile(ipv4_pattern)
ipv6_regex = regex.compile(ipv6_pattern)
ip_regex = regex.compile(ip_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
email_regex = regex.compile(email_pattern, flags=regex.MULTILINE|regex.VERBOSE) #, re.MULTILINE)
user_regex = regex.compile(user_pattern, flags=regex.MULTILINE|regex.VERBOSE) #, re.MULTILINE)
# phone_regex = regex.compile(phone_pattern, flags=regex.MULTILINE) #, re.MULTILINE)
# TODO: license


#sasha_regexes = copy.deepcopy(regex_rulebase)
mst_regexes = {}
for tag in high_risk_tags:
  #print(tag)
  if tag == 'ID':
    mst_regexes['ID'] = id_regex
  elif tag == 'KEY':
    mst_regexes['KEY'] = key_regex
  elif tag == 'IPv4':
    mst_regexes['IPv4'] = ipv4_regex
  elif tag == 'IPv6':
    mst_regexes['IPv6'] = ipv6_regex
  elif tag == 'IP_ADDRESS':
    mst_regexes['IP_ADDRESS'] = ip_regex
  elif tag == 'EMAIL':
    mst_regexes['EMAIL'] = email_regex
  elif tag == 'USER':
    mst_regexes['USER'] = user_regex
#  elif tag == 'NUMBER':
#    mst_regexes['NUMBER'] = phone_regex
  else:
    sys.stderr.write('Dont have tag regex pattern for %s =(' % tag)

#print("MST regexes under examination are:")
#for tag, regx in mst_regexes.items():
  #print(tag, end=":\t")
  #print(regx)

"""# PI Detection and Redaction functions are defined here! """

#@title The detection functions and basic filtering functions are defined here.
# tag_type = {'ID', 'KEY', 'EMAIL', 'IP_ADDRESS', 'PHONE', 'LICENSE_PLATE'}
# Choose whether to put this import before or after, depending on which you're testing. =)

def ip_has_digit(matched_str):
  """Checks to make sure the PII span is not just :: or whatever that may
  accidentally be picked up by making sure there are digits."""
  return any(map(str.isdigit, matched_str))

def matches_date_pattern(matched_str):
  # Screen out date false positives
  for year_regex in year_patterns:
    if year_regex.match(matched_str):
      return True
  return False

def is_website(matched_str):
  # TODO
  return False

def detect_pii(text, lang, tag_types):
  matches = []
  for tag in tag_types:
    label_pattern = mst_regexes[tag]
    # !! regex.match happens here!!
    matches_tmp = label_pattern.finditer(text)
    for match in matches_tmp:
      # TODO: Why does this happen?
      if match.groups():
        if len(match.groups()) > 1 and match.groups()[1]:
          sys.stderr.write("Warning: Found substring matches in the main match.")
          #print(tag)
          #print(text)
          #print(match.groups())
        matched_str = match.groups()
        # print(matched_str)
        # Why does this happen?
        matched_str = matched_str[0]
        if matched_str:
          if tag in ["IP_ADDRESS"]:
            # Filter out false positive IPs
            if not ip_has_digit(matched_str):
              continue
          if tag in ["ID", "IP_ADDRESS"]: #, "NUMBER"]:
            # Filter out date false positives
            if matches_date_pattern(matched_str):
              continue
          # TODO: Implement
          # if tag in ["KEY"]:
          #  # TODO: implement
          #  if is_website(matched_str):
          #    continue
          matches += [(matched_str, match.span(), str(label_pattern), tag, lang)]
  return matches


#@title Redaction function defined here.
def redact_pii(text, matches):
  """Takes a match as defined in the detect_pii function and redacts it from the full string, returning a <redacted text, metadata> tuple."""
  redacted_str = text
  metadata = []
  for match in matches:
    matched_str = match[0]
    tag = match[3]
    redact_tag = "PI:" + tag
    redacted_str = redacted_str.replace(matched_str, redact_tag)
    # Create the "metadata" as all of the information we had before redaction
    metadata += [(match)]
  return (redacted_str, metadata)

#@title General function to run the PII detection and redact it, saving everything else to metadata, is defined here.
def run_pii(text, lang):
  """
  Runs the given set of regexes on the data "lines" and pulls out the
  tagged items.
  The lines structure stores the language type(s). This can be used for
  language-specific regexes, although we're dropping that for now and using
  only "default"/non-language-specific regexes.
  """

  #print('Detecting....')
  # What is this for...?
  text = text.encode().decode()
  matches = detect_pii(text, lang, high_risk_tags)
  #print(matches)
  match_set = (text, {})
  if len(matches) > 0:
    # !!! REDACTION HAPPENS HERE !!!
    redacted_str, metadata = redact_pii(text, matches)
    metadata_out = {"regex metadata":metadata, "original": text, "redacted": redacted_str}
    match_set = (redacted_str, metadata_out)
  return match_set


def run_pii_batch(exs, lang):
    """
    Runs the given set of regexes on the data "lines" and pulls out the
    tagged items.
    The lines structure stores the language type(s). This can be used for
    language-specific regexes, although we're dropping that for now and using
    only "default"/non-language-specific regexes.
    """
    regex_metadata = []
    old_text = []
    new_text = []
    modified = []
    for text in exs["text"]:
        # What is this for...?
        text = text.encode().decode()
        matches = detect_pii(text, lang, high_risk_tags)
        if len(matches) > 0:
            # !!! REDACTION HAPPENS HERE !!!
            redacted_str, metadata = redact_pii(text, matches)
            regex_metadata.append(repr(metadata))
            old_text.append(text)
            new_text.append(redacted_str)
            modified.append(True)
        else:
            regex_metadata.append("")
            old_text.append(text)
            new_text.append(text)
            modified.append(False)
    result = {
        "regex_metadata": regex_metadata,
        "old_text": old_text,
        "text": new_text,
        "modified": modified
    }
    return result
