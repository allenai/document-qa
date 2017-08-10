import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"


def get_wiki_page_ids(article_titles, per_query=25):
    """
    Utility method to get the page_id and resolve re-directs for a set of wikipedia titles
    """
    wiki_page_ids = {}

    for i in range((len(article_titles) + per_query - 1) // per_query):
        start = i * per_query
        end = min((i + 1) * per_query, len(article_titles))
        original_titles = article_titles[start:end]

        r = requests.get(WIKI_API,
                         params=dict(action="query", format="json",
                                     redirects=True,
                                     titles="|".join(original_titles)))
        data = r.json()

        query = data["query"]
        if "redirects" in query:
            redirects = {x["to"]: x["from"] for x in query["redirects"]}
        else:
            redirects = {}

        for page, page_data in query["pages"].items():
            page = int(page)
            title = page_data["title"]

            if page == -1:
                raise ValueError()

            original_title = redirects.get(title, title)
            if original_title not in original_titles:
                raise ValueError(title)

            wiki_page_ids[original_title] = (title, page)

    return [wiki_page_ids[x] for x in article_titles]