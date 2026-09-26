# Fixed upstream baseline sources

`upstream/` contains only the five source files used by the CAA, CAST, LoReFT,
and RePS adapters, plus each repository's original license. These files are
byte-identical to the fixed upstream revisions in
`UPSTREAM_SOURCE_MANIFEST_V2.json`. The manifest also records the original full
tree/archive hashes; those large trees and archives are not part of this release.

| Baseline | Source repository | Frozen commit | License |
|---|---|---|---|
| CAA | https://github.com/nrimsky/CAA | `5dabbbd9a0bca5f25e174501e959de378806aa48` | MIT |
| CAST | https://github.com/IBM/activation-steering | `52be60235ee309b46c49d6d5877f36e20c52e6ab` | Apache-2.0 |
| LoReFT | https://github.com/stanfordnlp/pyreft | `dafd0995a366d7b47160a337dcc388eda7431821` | Apache-2.0 |
| RePS | https://github.com/stanfordnlp/axbench | `41c8332543e5a631f9a8c0a9df38799893ace758` | Apache-2.0 |

The paper evaluates task adaptations derived from these sources. These are not
unmodified reproductions of the original papers' evaluation protocols.
