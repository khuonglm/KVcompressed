# Visual-KV selection

Does unreliable text steer which visual cache entries a vision-language model keeps? Terms used across PREREG, the
code and the reports.

## Visual span

**Layout**:
The maps and newline tokens of one image's visual span, in span order.
_Avoid_: grid (except for one map), geometry

**Map**:
A rectangle of feature tokens inside the visual span, with its rows and columns.
_Avoid_: tile (for the map itself), grid

**Feature token**:
A visual-span token that carries image content; every token of the span except the newlines.
_Avoid_: patch (outside evidence-box mapping), visual entry

**Newline token**:
A token ending a row of a map; always kept and never counted in a budget.

## Selection

**Selector**:
A pre-registered rule that picks which feature tokens a unit keeps: pw, qspan, lt, uniform, random, restore or full.
_Avoid_: method, strategy, kind

**Validation kept-set kind**:
A shape of kept set used only to test the replay engine (random, concentrated, tail, scored), never a selector.
_Avoid_: selector

**Budget**:
How many feature tokens a unit keeps: an absolute K or a share.

**Share**:
A budget as a fraction of an image's feature tokens, which must give a whole number of tokens.
_Avoid_: ratio, percentage budget

**Kept set**:
The visual-span positions a unit keeps: its chosen feature tokens plus every newline token.
_Avoid_: selection (for the set itself), mask

## Protocol

**Stage**:
One freeze point of a study, which declares its inputs, settings and the PREREG section that must not be draft.
_Avoid_: step, phase

**Record**:
The file a stage writes once at its freeze, holding the hashes of its inputs, its settings and its code identity.
_Avoid_: freeze file, snapshot

**Amendment**:
A dated PREREG subsection that states a change after a freeze and why; it changes no frozen rule silently.
_Avoid_: revision, edit

**Amendment log**:
The list of accepted moves from a frozen hash to a new one, each dated, with its reason and evidence.
_Avoid_: changelog

**Code identity**:
The hash of the package's source files, by relative path, that a record freezes.
_Avoid_: code hash (for the concept), version

**Refusal**:
A stop that needs the researcher's action, such as a changed input or a draft section; not a crash.
_Avoid_: error (for this case), failure
