# Palette's Journal - UX & Accessibility Learnings

## 2024-05-23 - Improving Filter Interaction on Dashboard
**Learning:** Users often expect "Enter" to submit filters in a form-like interface even without an explicit `<form>` tag. Adding `onkeydown` listener for 'Enter' key on inputs bridges this gap in SPA-like dashboards where full page reloads are avoided.
**Action:** Always add keyboard listeners to filter inputs in AJAX-driven views and ensure visual feedback (loading state) on the trigger button.
