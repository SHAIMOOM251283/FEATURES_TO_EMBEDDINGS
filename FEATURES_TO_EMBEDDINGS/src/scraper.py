import asyncio
import csv
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BookScraper:

    def __init__(self):
        self.base_url = "https://books.toscrape.com/"
        data_dir = "data"
        self.output_dir = PROJECT_ROOT / data_dir
        file_name = "books_raw.csv"
        self.output_path = self.output_dir / file_name
        self.concurrency = 5
        self.headless = True
        self.block_assets = True
        self.rating_words = {
            "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
        }
        self.book_urls = []
        self.records = []
        self.failures = []
        self.browser = None
        self.context = None
        self.semaphore = None
        self.started_at = None

    async def block_heavy_requests(self, route):
        """Abort images, fonts and stylesheets — none of them carry data."""
        if route.request.resource_type in {"image", "font", "stylesheet", "media"}:
            await route.abort()
        else:
            await route.continue_()

    async def open_browser(self, playwright):
        """Launch one browser and one shared context for the whole run."""
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="FEATURES_TO_EMBEDDINGS/1.0",
            java_script_enabled=False,
        )
        if self.block_assets:
            await self.context.route("**/*", self.block_heavy_requests)
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def close_browser(self):
        """Tear down the context and browser."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()

    async def collect_book_urls(self):
        """Page through the catalogue sequentially and gather detail URLs."""
        page = await self.context.new_page()
        page_url = f"{self.base_url}catalogue/page-1.html"
        page_number = 1

        while page_url:
            try:
                await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as error:
                print(f"  catalogue page failed: {page_url} ({error})")
                break

            hrefs = await page.eval_on_selector_all(
                "article.product_pod h3 a",
                "nodes => nodes.map(n => n.href)",
            )
            self.book_urls.extend(hrefs)
            print(f"page {page_number}: {len(self.book_urls)} urls so far")

            next_href = await page.eval_on_selector_all(
                "li.next a", "nodes => nodes.map(n => n.href)"
            )
            page_url = next_href[0] if next_href else None
            page_number += 1

        await page.close()
        return self.book_urls

    def rating_from_classes(self, class_names):
        """Convert the star-rating CSS class list into an integer 1-5."""
        for css_class in class_names.split():
            if css_class in self.rating_words:
                return self.rating_words[css_class]
        return None

    def availability_from_text(self, text):
        """Pull the integer stock count out of text like 'In stock (19 available)'."""
        if not text:
            return None
        match = re.search(r"(\d+)\s+available", text)
        return int(match.group(1)) if match else 0

    def price_from_text(self, text):
        """Strip the currency symbol and return price as a float."""
        if not text:
            return None
        match = re.search(r"([\d.]+)", text)
        return float(match.group(1)) if match else None

    async def extract_fields(self, page, url):
        """Read every field off one loaded detail page in a single DOM pass."""
        raw = await page.evaluate(
            """() => {
                const pick = (sel) => {
                    const node = document.querySelector(sel);
                    return node ? node.textContent.trim() : null;
                };
                const table = {};
                document.querySelectorAll('table.table-striped tr').forEach(row => {
                    const header = row.querySelector('th');
                    const value = row.querySelector('td');
                    if (header && value) {
                        table[header.textContent.trim()] = value.textContent.trim();
                    }
                });
                const ratingNode = document.querySelector('article.product_page p.star-rating');
                const crumbs = document.querySelectorAll('ul.breadcrumb li a');
                const descHeading = document.querySelector('#product_description');
                let description = '';
                if (descHeading) {
                    let sibling = descHeading.nextElementSibling;
                    while (sibling && sibling.tagName !== 'P') {
                        sibling = sibling.nextElementSibling;
                    }
                    if (sibling) description = sibling.textContent.trim();
                }
                return {
                    title: pick('article.product_page h1'),
                    priceText: pick('article.product_page p.price_color'),
                    availabilityText: pick('article.product_page p.instock.availability'),
                    ratingClasses: ratingNode ? ratingNode.className : '',
                    category: crumbs.length >= 3
                        ? crumbs[crumbs.length - 1].textContent.trim() : null,
                    description: description,
                    upc: table['UPC'] || null,
                };
            }"""
        )

        return {
            "upc": raw["upc"],
            "title": raw["title"],
            "category": raw["category"],
            "price": self.price_from_text(raw["priceText"]),
            "rating": self.rating_from_classes(raw["ratingClasses"]),
            "availability": self.availability_from_text(raw["availabilityText"]),
            "description": raw["description"],
            "url": url,
        }

    async def scrape_book(self, url):
        """Open one page under the concurrency limit and return its record."""
        async with self.semaphore:
            page = await self.context.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return await self.extract_fields(page, url)
            except Exception as error:
                self.failures.append((url, str(error)))
                return None
            finally:
                await page.close()

    async def scrape_books(self):
        """Run every book URL through the scraper concurrently."""
        total = len(self.book_urls)
        done = 0

        tasks = [asyncio.create_task(self.scrape_book(url)) for url in self.book_urls]
        for finished in asyncio.as_completed(tasks):
            record = await finished
            if record:
                self.records.append(record)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"scraped {done}/{total}")

        return self.records

    def save(self):
        """Write all records to CSV."""
        if not self.records:
            print("nothing to save")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = ["upc", "title", "category", "price", "rating",
                      "availability", "description", "url"]
        with open(self.output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"saved {len(self.records)} records to {self.output_path}")

    def report(self):
        """Print a short summary so problems surface before modelling starts."""
        missing_description = sum(1 for r in self.records if not r["description"])
        missing_rating = sum(1 for r in self.records if r["rating"] is None)
        categories = {r["category"] for r in self.records if r["category"]}
        unique_upcs = {r["upc"] for r in self.records if r["upc"]}
        elapsed = time.time() - self.started_at

        print(f"\nrecords:              {len(self.records)}")
        print(f"unique upcs:          {len(unique_upcs)}")
        print(f"distinct categories:  {len(categories)}")
        print(f"empty descriptions:   {missing_description}")
        print(f"missing ratings:      {missing_rating}")
        print(f"failed pages:         {len(self.failures)}")
        print(f"elapsed:              {elapsed:.1f}s")

        for url, error in self.failures[:5]:
            print(f"  failure: {url} -> {error[:80]}")

    async def run_async(self):
        """Open the browser, collect URLs, scrape, save, then summarise."""
        self.started_at = time.time()
        async with async_playwright() as playwright:
            await self.open_browser(playwright)
            try:
                await self.collect_book_urls()
                await self.scrape_books()
            finally:
                await self.close_browser()
        self.save()
        self.report()
        return self.records

    def run(self):
        """Synchronous entry point."""
        return asyncio.run(self.run_async())


if __name__ == "__main__":
    BookScraper().run()