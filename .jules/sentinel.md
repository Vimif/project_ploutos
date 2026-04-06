## 2026-04-06 - Overly Permissive CORS Configuration
**Vulnerability:** The Flask dashboard had a globally permissive CORS configuration (`CORS(app)`) that allowed any origin to make cross-origin requests to the application.
**Learning:** Default implementations of CORS extensions often default to allowing all origins (`*`) if not explicitly restricted, which can expose API endpoints to Cross-Site Request Forgery (CSRF) or unauthorized cross-origin data access.
**Prevention:** Always restrict CORS policies to explicitly trusted origins (e.g., via environment variables) and scope the policy to only the endpoints that actually require cross-origin access (e.g., `r"/api/*"`).
