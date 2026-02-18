## 2026-02-18 - Filter Form Polish
**Learning:** Flask templates often use `onclick` handlers which can block the UI thread during async operations if not careful. Labels were missing `for` attributes, breaking accessibility.
**Action:** Replace `onclick` with `addEventListener` for better separation of concerns and testability. Ensure all labels have matching `for` attributes.
