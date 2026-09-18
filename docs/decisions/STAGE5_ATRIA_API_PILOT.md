# Atria API integration pilot — 2026-09-18

The owner authorized `Atria-Dawn-Preview` at
`https://api.atria-asi.ai/v1/responses`, using the existing local
`ATRIA_API_KEY`, and subsequently limited this round to 20 review entries.
This is an independent-judge integration test on already generated anonymous
answers. It is not new candidate generation or a formal full-set evaluation.

Selection is the first 20 entries in the frozen public packet: 17 base
comparisons and 3 position flips, covering 6 questions from 3 clinical cases.
The prefix is not representative of the full 408-question evaluation set.
The private model mapping and candidate hidden reasoning are never sent.
Every request explicitly disables tools. Credentials remain in process memory.

Three configurations were tried for technical compatibility:

1. `s5_atria_dawn_20260918T081010Z`: `temperature=0` and native
   `text.format.type=json_object`. The first response was HTTP 200 / completed
   but contained malformed JSON. It is retained and excluded from scores.
2. `s5_atria_dawn_20260918T081113Z`: same prompt, plain text transport,
   `temperature=0`. The first entry returned a valid judgment. The next entry
   returned repeated `/0` text despite HTTP 200 / completed. Both are retained.
3. `s5_atria_dawn_20260918T081417Z`: omit temperature, as in the owner's
   minimal example; retain the same prompt, token cap and no-tool settings.
   Continue from entry index 2. Do not regenerate the valid first judgment or
   retry the second entry. Record invalid outputs and stop after three
   consecutive invalid responses or a transport failure.

There was one explicit technical retry of the first entry, so completing the
20-entry test can require 21 scoring requests plus the initial `hi` connection
probe. No full-set run is authorized. This pilot compares API compatibility,
not clinical correctness or model rankings. Configuration changes were made
after malformed outputs, not in response to model scores. They do not prove
that temperature or JSON mode caused the observed failures.

Retain protocol, runner source snapshot, command, code revision, Python/client
environment, exact request body without authentication, raw response, token
usage, elapsed request time, validation result and normalized judgment where
valid. Scores are grouped by judge configuration; invalid outputs are never
assigned zero or repaired into invented judgments. Preview alias provenance
is limited to the returned model identity and request/timestamp evidence.
No currency estimate is available without provider pricing/billing records.

An API judge does not satisfy the real human-audit requirement. All critical
flags must enter that queue. Stage5 remains incomplete and Stage6 stays pending.
