## Strategy to get depth 1

### Context

Once we've extract all the seed pages, we plan to make a pseudo crawl. The idea is simple:
 - we extract the outgoing urls from those pages.
 - we find the most recent record in CC matching that url (if it exists).
 - we do the entire processing for all the new records.pages
 - we update `outgoing_urls` to obtain `outgoing_ids`

### Process

 - 1) Make Athena query
 - 2) Preprocess dataset to: load_warc, obtain pdf_urls, extract external_urls
 - 3) Build new query with all `external_urls`
 - 4) Repeat 1-3 until reaching the depth we want.
 - 5) Finalise `finalise.py` to: generate ids, generate `external_ids` that map to rows inside dataset.
