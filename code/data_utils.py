from requests import post, get
from dotenv import load_dotenv
import os
import time
import json
import base64
import numpy as np
import pandas as pd

#loads env vars located in .env file
load_dotenv()

#get and print client id and secret
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
print(CLIENT_ID, CLIENT_SECRET)

ROOT_API_URL = "https://api.spotify.com/v1"




def str_list_to_array(str_list):
    return np.array([json.loads(v) for v in str_list])
    
def xdict(d):
    return dict(d or {})

def xlist(d):
    return (d or [])
    
def get_token():
    auth_string = CLIENT_ID+":"+CLIENT_SECRET
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url="https://accounts.spotify.com/api/token"
    headers= {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    js_result = json.loads(result.content)
    token=js_result["access_token"]
    return token


def get_auth_header(token):
    return {"Authorization": "Bearer " + token}


def search_artist(token, artist_name):
    url = os.path.join(ROOT_API_URL, "search")
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"

    query_url = url + query
    result = get(query_url, headers=headers)
    js_result = json.loads(result.content)['artists']['items']
    if len(js_result)==0:
        print("No artist with this name exists...")
        return None
    return js_result[0]

def get_tracks_by_artist(token, artist_id):
    url = os.path.join(ROOT_API_URL, f"artists/{artist_id}/top-tracks?country=US")
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    js_result = json.loads(result.content)['tracks']
    return js_result

def get_playlist(token, playlist_id, offset=0):
    url = os.path.join(ROOT_API_URL, f"playlists/{playlist_id}/tracks?offset={offset}")
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    js_result = json.loads(result.content)
    return js_result

def get_artist_albums(token, artist_id, offset=0, limit=None):
    if limit is not None:
        url = os.path.join(ROOT_API_URL, f"artists/{artist_id}/albums?offset={offset}&limit={limit}")
    else:
        url = os.path.join(ROOT_API_URL, f"artists/{artist_id}/albums?offset={offset}")
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    js_result = json.loads(result.content)
    return js_result['items']

def get_album_tracks(token, album_id, offset=0, limit=None):
    if limit is not None:
        url = os.path.join(ROOT_API_URL, f"albums/{album_id}/tracks?offset={offset}&limit={limit}")
    else:
        url = os.path.join(ROOT_API_URL, f"albums/{album_id}/tracks?offset={offset}")
    headers = get_auth_header(token)
    result = get(url, headers=headers)
    js_result = json.loads(result.content)
    return js_result['items']
    
def get_playlist_tracks(token, playlist_id, summarize=False):
    tracks_info = []
    playlist_info = get_playlist(token, playlist_id, offset=0)
    n_tracks = playlist_info['total']
    tracks_info += playlist_info['items']
    offset=1
    while n_tracks>(offset*100):
        print(offset)
        print('offset: ', offset*100)
        playlist_info = get_playlist(token, playlist_id, offset=int(offset*100))
        tracks_info += playlist_info['items']
        offset+=1
    if summarize:
        track_id = [t['track']['id'] for t in tracks_info]
        track_name = [t['track']['name'] for t in tracks_info]
        select_track_keys = ['name', 'id', 'popularity']
        select_artist_keys = ['name', 'id']
        track_ls = [{'track_%s'%k: t['track'][k] for k in select_track_keys} for t in tracks_info]
        artist_ls = [{'artist_%s'%k: t['track']['artists'][0][k] for k in select_artist_keys} for t in tracks_info]
        
        tracks_info = pd.DataFrame([{**a, **t} for a, t in zip(artist_ls, track_ls)])
        
    return tracks_info

def get_track_name_id(tracks_info):
    track_names = []
    track_ids = []
    for track in tracks_info:
        track_names.append(track['track']['name'])
        track_ids.append(track['track']['id'])
    return track_names, track_ids

def get_audio_features(token, track_ids):
    delta=50
    chunked_track_ids = [track_ids[x:x+delta] for x in range(0, len(track_ids), delta)]
    delim = '%2C'
    headers = get_auth_header(token)
    # https://api.spotify.com/v1/audio-features?ids=7ouMYWpwJ422jRcDASZB7P%2C4VqPOruhp5EdPBeR92t6lQ%2C2takcwOaAZWiXQijPHIx7B
    sleep_min = 1
    sleep_max = 5
    start_time = time.time()
    track_count = 0
    audio_features_ls = []
    for i, track_set in enumerate(chunked_track_ids):
        track_count += delta
        track_query_str = delim.join(track_set)
        url = f'https://api.spotify.com/v1/audio-features?ids={track_query_str}'
        # print(url)
        result = get(url, headers=headers)
        js_result = json.loads(result.content)['audio_features']
        audio_features_ls += js_result
        print(len(audio_features_ls))
        print(f'request {i+1} - {len(audio_features_ls)} tracks completed')
        r_val = np.random.uniform(sleep_min, sleep_max)
        if (i+1) < len(chunked_track_ids):
            print(f'Next request in ~ {int(np.round(r_val))} seconds')
            time.sleep(r_val)
        print('Elapsed Time: {} seconds'.format(time.time() - start_time))
    return [dict(reversed(list(xdict(t).items()))) for t in audio_features_ls]
    # if 'items' in audio_features_ls[0].keys():
    #     return [dict(reversed(list(t.items()))) for t in audio_features_ls]
    # else:
    #     return [t for t in audio_features_ls]

def get_audio_analysis(token, track_ids, track_names=None, sleep_range=(1,5)):
    headers = get_auth_header(token)
    sleep_min = sleep_range[0]
    sleep_max = sleep_range[1]
    start_time = time.time()
    track_count = 0
    audio_analysis_ls = []
    for i, t_id in enumerate(track_ids):
        url = f'https://api.spotify.com/v1/audio-analysis/{t_id}'
        result = get(url, headers=headers)
        js_result = json.loads(result.content)['segments']
        print(len(audio_analysis_ls))
        
        audio_analysis_df = pd.DataFrame(js_result)
        audio_analysis_df.insert(2, 'end', audio_analysis_df['start']+audio_analysis_df['duration'])
        attack_aggression = (audio_analysis_df['loudness_max']-audio_analysis_df['loudness_start']) / audio_analysis_df['loudness_max_time']
        audio_analysis_df.insert(8, 'attack_aggression', attack_aggression)
        #attack_aggression = (loudness_max - loudness_start) / loudness_max_time -- higher ~ more aggressive attack
        #pitch ~ frequency spectrum -- 12 basis coefficients
        #loudness_max ~ amplitude 
        #timbre -- 12 basis coefficients -- x:time y:pitch z:loudness_max
        audio_analysis_ls += [audio_analysis_df]
        print(f'request {i+1} - {len(audio_analysis_ls)} tracks completed')
    
        r_val = np.random.uniform(sleep_min, sleep_max)
        if (i+1) < len(track_ids):
            print(f'Next request in ~ {int(np.round(r_val))} seconds')
            time.sleep(r_val)
        print('Elapsed Time: {} seconds'.format(time.time() - start_time))

    if track_names is not None:
        for t_id, t_name, audio_df in zip(track_ids, track_names, audio_analysis_ls):
            audio_df.insert(0, 'track_id', t_id)
            audio_df.insert(0, 'track_name', t_name)
    else:
        for t_id, t_name, audio_df in zip(track_ids, audio_analysis_ls):
            audio_df.insert(0, 'track_id', t_id)
            
    audio_analysis_df = pd.concat(audio_analysis_ls, axis=0).reset_index(drop=True)
    return audio_analysis_df
    
def get_audio_analysis_item(analysis_df, items = ['timbre', 'pitches']):
    if isinstance(items, str):
        # If the input is a string, convert it into a list containing the string
        items = [items]
    if not isinstance(items, list):
        # If the input is neither a string nor a list, raise an error or handle it as needed
        raise ValueError("input-items must be a string or a list")
    if not set(items).issubset(set(['pitches', 'timbre'])):
        raise ValueError("function currently only supports getting timbre and pitches")
        
    track_ids = analysis_df.track_id.unique()
    item_dict={i: [] for i in items}
    for track_id in track_ids:
        for item in items:
            track_item = analysis_df[analysis_df.track_id==track_id][item].values.tolist()
            track_item = str_list_to_array(track_item)
            item_dict[item] += [track_item]
    return item_dict




def xlist(x):
    return (x or [])

def xtolist(x):
    #handle None type
    x = xlist(x)

    # Check if x is a string
    if isinstance(x, str):
        # Convert the string to a list containing the string
        x = [x]
    elif isinstance(x, list):
        # Check if each item in the list is a string, if not, raise an error
        if not all(isinstance(item, str) for item in x):
            raise ValueError("Each item in the input list must be a string")
    else:
        # If the input is neither a string nor a list, raise an error or handle it as needed
        raise ValueError("input-x must be a string or a list of strings")

    return x


def get_recs(token, seed_tracks=None, seed_artists=None, limit=None, max_popularity=None):
    seed_tracks = xtolist(seed_tracks)
    seed_artists = xtolist(seed_artists)

    if not seed_tracks and not seed_artists:
        raise ValueError("Must provide at least one seed track or seed artist")

    headers = get_auth_header(token)
    query_url = "https://api.spotify.com/v1/recommendations?"
    if seed_tracks:
        query_url += f"seed_tracks={'%2C'.join(seed_tracks)}"
    if seed_artists:
        query_url += f"&seed_artists={'%2C'.join(seed_artists)}"
    if isinstance(limit, int):
        query_url += f"&limit={limit}"
    if isinstance(max_popularity, int):
        query_url += f"&max_popularity={max_popularity}"

    result = get(query_url, headers=headers)
    js_result = json.loads(result.content)['tracks']
    artists = [res['artists'][0]['name']  for res in js_result]
    release_dates = [res['album']['release_date'] for res in js_result]
    df_track_recs = pd.DataFrame(js_result)[['name', 'popularity']]
    df_track_recs.insert(0, 'artist', artists)
    df_track_recs.insert(0, 'release_date', release_dates)
    df_track_recs['query'] = [f"artist:\042{a}\042 track:\042{t}\042" for a,t in zip(df_track_recs['artist'].tolist(), df_track_recs['name'].tolist())]
    return df_track_recs