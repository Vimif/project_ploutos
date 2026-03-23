## 2026-03-23 - Interactive Form Labels
**Learning:** In the dashboard forms, some `<label>` elements inside `.filter-group` classes were not explicitly linked to their corresponding inputs/selects via the `for` attribute, which impacts screen reader accessibility and prevents users from focusing the input by clicking the label.
**Action:** Ensure all `<label>` tags have a `for` attribute that correctly points to the `id` of their input element, and include `cursor: pointer` to indicate that clicking them works.
