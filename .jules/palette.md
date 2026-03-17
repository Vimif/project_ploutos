## 2024-03-17 - Filter Group Accessibility
**Learning:** Dashboard form elements within `filter-group` containers must explicitly use `for` attributes on `<label>` tags linked to the `id` of their respective `input` or `select` elements, and include `cursor: pointer` to indicate interactivity. Elements without explicit text (like icon buttons) must use `aria-label` to ensure screen reader accessibility.
**Action:** Always verify form labels explicitly point to their input counterparts using the `for` attribute and have proper cursor styles for clear visual feedback.
