# What

`create_embeddings.py` takes a JSON file from [mtgjson.com](mtgjson.com) and creates embeddings via the [nomic.ai](https://blog.nomic.ai/posts/nomic-embed-text-v1) model. The results are stored in a parquet file, and can then be read for further analysis.

`find_similar.py` takes either an exact card name or a free form text query (these do not work very well, but can be interesting) and finds the top N most similar cards in the database.

`most_odd.py` is a quick attempt at finding the most unique cards, i.e. the cards that are least similar to other cards. This should work well given a filtered data set, on the [AtomicCards](https://mtgjson.com/data-models/card/card-atomic/) set that contains every unique card it somewhat unsurprisingly mostly finds lands, creatures without any text, and schemes. 

# How

First, install all the relevant requirements into a new virtual environment. Python 3.11 is recommended. On OS X, this is trivially installed via `homebrew`. Then, download a data set from [mtgjson.com](mtgjson.com) and, if necessary, extract the JSON file. Then, create the embeddings:

```
python3.11 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python3 create_embeddings.py
python3 create_embeddings.py --json StandardAtomic.json --parquet StandardAtomic.parquet
```

On a Macbook M1, creating embeddings for Standard Atomic takes about 4:30 minutes, while the set of all cards takes about 40 minutes.

Then, query away:
```
$ time python3 find_similar.py --card 'Lightning Bolt' --parquet AtomicCards.parquet --top 10
Top 10 items similar to 'Lightning Bolt':
Name: Lightning Strike, Similarity: 0.9632233907976071
Name: Lightning Blast, Similarity: 0.9578671652885031
Name: Lightning Dart, Similarity: 0.9269876080609935
Name: Radiating Lightning, Similarity: 0.9267595927275938
Name: Electrostatic Bolt, Similarity: 0.9125906774927794
Name: Rhystic Lightning, Similarity: 0.9109785167018958
Name: Homing Lightning, Similarity: 0.9104548822040253
Name: Blastfire Bolt, Similarity: 0.9048170577510474
Name: Lightning Helix, Similarity: 0.9010020957301721
Name: Twin Bolt, Similarity: 0.8958958009798943

real	0m1.742s
user	0m3.230s
sys	0m2.120s
$
```

# TODOs
- Add filters when querying, which would require amending the parquet file with extra fields for things like type and set legality
- Add an option to bring in the original JSON file for a verbose mode to print card details instead of just the name