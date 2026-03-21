## 2024-03-21 - Adding for attribute to form filters
**Learning:** Found an accessibility issue pattern where `filter-group` input and select elements lacked `for` attributes on their `<label>`s.
**Action:** Always link `<label>` elements to their associated inputs using the `for` attribute for screen reader compatibility and click-to-focus functionality.
