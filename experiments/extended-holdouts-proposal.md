# Extended holdout proposal (awaiting approval)

Criteria: full-Bible coverage, one language per target family bucket
(Bantu, Austronesian, Indo-Aryan, Turkic, + one other), Genesis withheld.
The `script` column comes from corpus metadata and is advisory —
spot-check the actual text before approving.

| languageCode   | translationId   | languageNameInEnglish   | family        | script     |   OTverses |   NTverses | heldOutBook   |
|:---------------|:----------------|:------------------------|:--------------|:-----------|-----------:|-----------:|:--------------|
| lin            | lin             | Lingála                 | Bantu         | Latin      |      23212 |       7957 | GEN           |
| pon            | pon2006a        | Pohnpeian               | Austronesian  | Latin      |      22266 |       7941 | GEN           |
| hin            | hin2017         | Hindi                   | Indo-Aryan    | Devanagari |      23145 |       7959 | GEN           |
| azb            | azb             | Azerbaijani, South      | Turkic        | Latin      |      23145 |       7927 | GEN           |
| vie            | vie1934         | Vietnamese              | Austroasiatic | Latin      |      23145 |       7957 | GEN           |

**To approve**: add each `translationId: [GEN]` entry to the
`holdouts:` map in `configs/holdouts.yaml` (edit picks freely first).
No training run may start until this is done (spec.md gate).
