This repository maintains brief summary of my readings. Most of theme are papers, but also includes remarkable things other than paper.

## Papers

- Sharma et al., Conceptual Captions, A Cleaned, Hypernymed, Image Alt-text Dataset for Automatic Image Captionning, ACL 2018.
  - Presented a dataset of image caption annotations, called Conceptual Captions, which consists of 3.3M <image, description> pairs.
  - Used a Flume pipeline to extract, filter, and processes candidate <image, caption> pairs.
    - Image-based filtering: discards images based on encoding format, size, aspect ratio, and offensive content. Only keeps JPEG images where both dimensions are greater than 400 pixels, and the ratio of large to smaller dimension is no more than 2. Excludes images that trigger pornography or profanity detectors.
    - Text-based filtering: harvests Alt-text from HTML webpages. Analyzed candidate Alt-text using the Google Cloud Language APIs, including part-of-speech (POS), sentiment/polarity, and pornography/profanity annotations. Candidates with no determiner, no noun, or no preposition are discarded. Candidates with a nigh noun ratio are also discarded. Candidates with a high rate of token repetition are discarded. Candidates where the first word is not capitalized, or with too high capitalized-word ratio are discarded. Discarded candidates that contain tokens that are not appearing in the English Wikipedia at least 5 times. Candidates that score too high or too low on the polarity annotations, or trigger the pornography/profanity detectors, are discraded. Predefined boiller-plate prefix/suffix sequences matching the text are croppsed, e.g., "click to enlarge picture", "stock photo". Also drop text which begins/ends in certain patterns, e.g., "embedded image parmalink", "profile photo".
    - Image&Text-based filtering: filter out candidates for which none of the text tokens can be mapped to the content of the image.
    - Text transformation with hypernymization: noun modifiers of certain types (proper nouns, number, units) are removed; dates, durations, and preposition-based locations (e.g., "in Los Angeles") are removed; named-entities are identified, matched against the KG entries, and substitute with their hypernym; resulting coordination noun-phrases with the same head (e.g., "actor and actor") are resolved into a single-head, pluralized form (e.g., "actors"). Too short or inconsistent samples are discarded after transformation. Cluster all resolved entities (e.g., "actor", "dog", "neighborhood") and keep only candidates for which all detected types have a count of over 100.

## Articles

## GiHub Repositories
