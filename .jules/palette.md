## 2026-02-15 - Filter Group Accessibility
**Learning:** Filter components (`.filter-group`) across this app consistently miss the `for` attribute linking labels to their `<select>` or `<input>` fields. This breaks screen reader support and creates frustratingly small click targets.
**Action:** Always add explicit `for="[id]"` attributes to `<label>` tags and add `cursor: pointer` to indicate interactivity when building or modifying form/filter layouts.
