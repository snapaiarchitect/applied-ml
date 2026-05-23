# Data Dictionary

## Dataset: 20 Newsgroups
**Source**: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)  
**Original Paper**: Lang, K. (1995). "Newsweeder: Learning to filter netnews." ICML 1995.  
**Last Updated**: 1995 (static benchmark dataset)  
**Update Frequency**: N/A — academic benchmark  

---

## Files

| File | Format | Records | Description |
|------|--------|---------|-------------|
| `20newsgroups_train.json` | JSON | 11,314 | Training documents with text + labels |
| `20newsgroups_test.json` | JSON | 7,532 | Test documents with text + labels |
| `20newsgroups_stats.csv` | CSV | 20 rows | Per-category document counts |

---

## Schema

### `20newsgroups_train.json` / `20newsgroups_test.json`

| Field | Type | Description |
|-------|------|-------------|
| `data` | list[str] | Raw document text (including headers) |
| `target` | list[int] | Category index (0–19) |
| `target_names` | list[str] | Category labels (20 newsgroup names) |
| `filenames` | list[str] | Original file paths in corpus |
| `DESCR` | str | Dataset description |

### Categories (20)

| Index | Category | Description |
|-------|----------|-------------|
| 0 | `alt.atheism` | Atheism discussions |
| 1 | `comp.graphics` | Computer graphics |
| 2 | `comp.os.ms-windows.misc` | Windows OS |
| 3 | `comp.sys.ibm.pc.hardware` | IBM PC hardware |
| 4 | `comp.sys.mac.hardware` | Mac hardware |
| 5 | `comp.windows.x` | X Window System |
| 6 | `misc.forsale` | Classified ads |
| 7 | `rec.autos` | Automobiles |
| 8 | `rec.motorcycles` | Motorcycles |
| 9 | `rec.sport.baseball` | Baseball |
| 10 | `rec.sport.hockey` | Hockey |
| 11 | `sci.crypt` | Cryptography |
| 12 | `sci.electronics` | Electronics |
| 13 | `sci.med` | Medicine |
| 14 | `sci.space` | Space |
| 15 | `soc.religion.christian` | Christianity |
| 16 | `talk.politics.guns` | Gun politics |
| 17 | `talk.politics.mideast` | Middle East politics |
| 18 | `talk.politics.misc` | General politics |
| 19 | `talk.religion.misc` | Religion |

---

## Data Quality Notes

- Real Usenet posts from the 1990s — includes headers, signatures, quoted text
- Headers contain metadata (From, Subject, Organization) that can leak labels
- Some categories are closely related (`comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`) making them harder to distinguish
- `sci.electronics` and `sci.crypt` share technical jargon that confuses classifiers

---

## API vs. Fallback

| Scenario | Behavior |
|----------|----------|
| sklearn available | `src/fetch_20newsgroups.py` uses `sklearn.datasets.fetch_20newsgroups()` |
| sklearn unavailable | Falls back to local JSON cache in `data/raw/` |
| Internet down | Uses pre-saved `data/raw/20newsgroups_*.json` files |

---

## Citation

Lang, K. (1995). "Newsweeder: Learning to filter netnews." In *Proceedings of the 12th International Conference on Machine Learning (ICML 1995)*, pp. 331–339.
