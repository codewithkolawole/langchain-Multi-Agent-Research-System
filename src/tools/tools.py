from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re



load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str)->str:
    """" Search the web for recent and reliable information on a topic.Returns Title"""
    results = tavily.search(query=query,max_results=4)

    out = []
    for result in results["results"]:
        out.append(
            f"Title:{result["title"]}\nURL : {result["url"]}\nSnippet:{result['content'][:300]}\n"
        )
    return "\n-----\n".join(out)



@tool
def scrape_url(url:str)->str:
    """
    Scrapes the web for recent and reliable information on a topic
    use multiple extraction strategies for better reliability
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.1 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com",
    }

    try:
        response = requests.get(url, headers=headers,timeout=15)
        response.raise_for_status()

        html = response.text


        #strategy 1 ->trafilatura (best for articles/blogs)
        extracted = trafilatura.extract(html,include_comments=False,include_tables=False)

        if extracted and len(extracted.strip())>200:
            cleaned = re.sub(r"\s+"," ",extracted)
            return cleaned[:5000]


        #strategy 2 readability
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form"
        ]):
            tag.decompose()
        text = soup.get_text(separator=" ",strip=True)

        if text and len(text.strip())>200:
            cleaned = re.sub(r"\s+"," ",text)
            return cleaned[:5000]

        #strategy 3 fallback full page extraction
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup([
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ",strip=True)
        cleaned = re.sub(r"\s+"," ",text)

        if cleaned:
            return cleaned[:5000]

        return "couldn't extract meaningful information"
    except requests.exceptions.Timeout:
        return "Return Url timed out while scarping"

    except requests.exceptions.HTTPError as e:
        return f"Http error occurred: {str(e)}"

    except Exception as e:
        return f"Could not Scrape URL: {str(e)}"



