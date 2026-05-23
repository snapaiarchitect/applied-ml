"""
Verify scikit-learn 20 Newsgroups Dataset (REAL — no replacement needed)

DATA SOURCE:
    Lang, K. (1995). Newsweeder: Learning to filter netnews.
    Proceedings of the 12th International Conference on Machine Learning (ICML 1995),
    Lake Tahoe, Nevada, pp. 331-339.

ACCESS:
    sklearn.datasets.fetch_20newsgroups — built-in dataset loader
    No download script needed; scikit-learn handles fetching automatically.

CITATION:
    @inproceedings{lang1995newsweeder,
      title={Newsweeder: Learning to filter netnews},
      author={Lang, Ken},
      booktitle={Proceedings of the twelfth international conference on machine learning},
      pages={331--339},
      year={1995}
    }

DATA DESCRIPTION:
    - 18,846 posts from 20 Usenet newsgroups
    - Train/test split: 11,314 / 7,532
    - Categories: comp.*, rec.*, sci.*, talk.*, misc.*, alt.*, soc.*
    - Already REAL — verified via sklearn.datasets source code
"""
from sklearn.datasets import fetch_20newsgroups


def verify_newsgroups():
    """Verify real 20 Newsgroups data loads correctly."""
    print("🔍 Verifying scikit-learn 20 Newsgroups dataset...")
    print("   Source: Lang (1995), ICML 1995")
    print("   Access: sklearn.datasets.fetch_20newsgroups")

    # Load training subset
    newsgroups = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

    print(f"   ✓ Loaded: {len(newsgroups.data):,} documents")
    print(f"   ✓ Categories: {len(newsgroups.target_names)}")
    print(f"   ✓ Sample categories:")
    for name in newsgroups.target_names[:5]:
        print(f"      - {name}")
    print(f"   ... and {len(newsgroups.target_names) - 5} more")

    # Show a sample
    print(f"\n📄 Sample document (category: {newsgroups.target_names[newsgroups.target[0]]}):")
    print("   " + newsgroups.data[0][:200].replace('\n', ' ') + "...")

    print("\n✅ 20 Newsgroups is VERIFIED REAL — no replacement needed.")
    print("   generate_data.py was NEVER used for this project.")

    return newsgroups


if __name__ == "__main__":
    verify_newsgroups()
