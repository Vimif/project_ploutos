## 2024-03-20 - Adding interactive cursor and for attribute to filters
**Learning:** Found an accessibility issue pattern where `filter-group` input and select elements lacked `for` attributes on their `<label>`s, and did not have `cursor: pointer` to indicate they are interactive.
**Action:** Always link `<label>` elements to their associated inputs using the `for` attribute and ensure interactive elements indicate their state using `cursor: pointer`.
