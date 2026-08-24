"""Optional YouTube learning-resource lookup for missing skills."""
import requests

from modules import config

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def get_learning_videos(skills, max_skills=6):
    """Return one top tutorial video per skill.

    Degrades gracefully: if no API key or the request fails, returns [].
    Each item: {"skill", "title", "url", "thumbnail", "channel"}.
    """
    if not config.youtube_enabled() or not skills:
        return []

    videos = []
    for skill in skills[:max_skills]:
        params = {
            "part": "snippet",
            "q": f"learn {skill} tutorial for beginners",
            "type": "video",
            "maxResults": 1,
            "relevanceLanguage": "en",
            "safeSearch": "strict",
            "key": config.YOUTUBE_API_KEY,
        }
        try:
            resp = requests.get(_SEARCH_URL, params=params, timeout=15)
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError):
            continue

        if data.get("error"):
            # Quota exhausted or bad key — stop trying, fail silently.
            break

        items = data.get("items") or []
        if not items:
            continue

        item = items[0]
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id:
            continue

        thumbs = snippet.get("thumbnails") or {}
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")

        videos.append(
            {
                "skill": skill,
                "title": snippet.get("title", f"Learn {skill}"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": thumb,
                "channel": snippet.get("channelTitle", ""),
            }
        )
    return videos
