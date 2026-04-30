from playwright.sync_api import sync_playwright

def run_cuj(page):
    page.goto("http://127.0.0.1:5000")
    page.wait_for_timeout(500)

    # Check the main nav area
    nav_locator = page.locator('nav[aria-label="Main Navigation"]')
    nav_locator.wait_for()

    # Click first link
    page.get_by_role("link", name="Overview").click()
    page.wait_for_timeout(500)

    # Click second link
    page.get_by_role("link", name="Session Analysis").click()
    page.wait_for_timeout(500)

    # Take screenshot
    page.screenshot(path="/home/jules/verification/screenshots/verification.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    import os
    os.makedirs("/home/jules/verification/videos", exist_ok=True)
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
