import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError



load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

PROGRESS_PATH = os.path.join(DATA_DIR, "progress.json")
TMP_PROGRESS_PATH = os.path.join(DATA_DIR, "progress_tmp.json")
OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "raw_youtube_data_v2.csv")


def get_client():
    return build(
        "youtube",
        "v3",
        developerKey=YOUTUBE_API_KEY
    )


def safe_execute(request, retries=3):

    for i in range(retries):

        try:
            return request.execute()

        except HttpError as e:

            error = str(e)

            if "Quota exceeded" in error:
                print("API quota exceeded.")
                return None

            print("API error:", e)

            time.sleep(2 ** i)


        except Exception as e:

            print("Unexpected error:", e)
            time.sleep(2 ** i)


    return None



def load_progress():

    if os.path.exists(PROGRESS_PATH):

        try:
            with open(PROGRESS_PATH, "r") as f:
                return json.load(f)

        except:
            pass

    return {
        "done": [],
        "seen": []
    }



def save_progress(done, seen):

    with open(TMP_PROGRESS_PATH, "w") as f:

        json.dump(
            {
                "done": done,
                "seen": list(seen)
            },
            f
        )

    os.replace(
        TMP_PROGRESS_PATH,
        PROGRESS_PATH
    )



def get_channel_subscribers(youtube, channel_ids):

    result = {}

    if not channel_ids:
        return result


    request = youtube.channels().list(
        part="statistics",
        id=",".join(channel_ids[:50])
    )


    response = safe_execute(request)


    if not response:
        return result


    for item in response.get("items", []):

        cid = item["id"]

        subs = (
            item.get("statistics", {})
            .get("subscriberCount")
        )

        result[cid] = subs


    return result




def fetch_videos(
        youtube,
        category_id,
        seen_ids,
        done_categories,
        limit=3000):


    results = []

    page_token = None


    keywords = [
        "vlog",
        "review",
        "tutorial",
        "news",
        "how to",
        "guide",
        "tips",
        "trending",
        "highlights",
        "commentary"
    ]


    index = 0



    while len(results) < limit:


        request = youtube.search().list(

            part="snippet",

            type="video",

            q=keywords[index % len(keywords)],

            videoCategoryId=category_id,

            maxResults=50,

            pageToken=page_token

        )


        response = safe_execute(request)


        if not response:
            break



        video_ids = []
        channel_ids = []


        for item in response.get("items", []):

            vid = item["id"].get("videoId")


            if vid and vid not in seen_ids:

                video_ids.append(vid)
                seen_ids.add(vid)



        if video_ids:


            details = youtube.videos().list(

                part="snippet,statistics,contentDetails",

                id=",".join(video_ids)

            )


            detail_response = safe_execute(details)


            if not detail_response:
                break



            for video in detail_response["items"]:

                channel_ids.append(
                    video["snippet"]["channelId"]
                )



            subscribers = get_channel_subscribers(
                youtube,
                channel_ids
            )



            batch=[]


            for video in detail_response["items"]:


                snippet = video["snippet"]

                stats = video["statistics"]

                content = video["contentDetails"]



                batch.append({

                    "video_id": video["id"],

                    "title": snippet.get("title",""),

                    "tags": ",".join(
                        snippet.get("tags",[])
                    ),

                    "description":
                        snippet.get("description",""),

                    "view_count":
                        stats.get("viewCount"),

                    "subscriber_count":
                        subscribers.get(
                            snippet["channelId"]
                        ),

                    "duration_seconds":
                        content.get("duration"),

                    "category_id":
                        category_id

                })



            results.extend(batch)



            pd.DataFrame(batch).to_csv(

                OUTPUT_CSV_PATH,

                mode="a",

                header=not os.path.exists(
                    OUTPUT_CSV_PATH
                ),

                index=False

            )



        page_token = response.get(
            "nextPageToken"
        )


        if not page_token:

            index += 1
            page_token=None


        if len(results) >= limit:
            break


        time.sleep(1)



    return results





if __name__ == "__main__":


    youtube = get_client()


    categories = [

        "1",
        "10",
        "15",
        "17",
        "20",
        "23",
        "24",
        "26"

    ]



    state = load_progress()


    done = state["done"]


    seen = set()



    if os.path.exists(OUTPUT_CSV_PATH):

        df = pd.read_csv(
            OUTPUT_CSV_PATH
        )

        seen = set(
            df["video_id"]
        )



    print("Starting collection...")



    for category in categories:


        if category in done:

            continue



        print(
            f"Collecting category {category}"
        )



        fetch_videos(

            youtube,

            category,

            seen,

            done,

            3000

        )



        done.append(category)

        save_progress(
            done,
            seen
        )



    print("Collection finished.")



    if os.path.exists(OUTPUT_CSV_PATH):

        df = pd.read_csv(
            OUTPUT_CSV_PATH
        )

        df.drop_duplicates(
            "video_id",
            inplace=True
        )


        df.to_csv(
            OUTPUT_CSV_PATH,
            index=False
        )


        print(
            "Final rows:",
            len(df)
        )