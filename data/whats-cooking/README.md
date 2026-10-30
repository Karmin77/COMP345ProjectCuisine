# What's Cooking (Yummly / Kaggle)

Labeled cuisine classification from ingredient lists.

- **Source competition:** https://www.kaggle.com/c/whats-cooking
- **Mirror used (labeled JSON):** https://github.com/RPI-Analytics/MGMT6963-2015/tree/master/data/whatscooking
- **Files:** `train.json` (39,774 labeled recipes), `test.json` (9,944 recipes, **no cuisine labels**)

Use **only `train.json`** for the COMP 345 experiment. The Kaggle test split cannot be scored locally. Make a stratified train/val/test split from `train.json` and freeze it.

Each train record:

```json
{"id": 10259, "cuisine": "greek", "ingredients": ["romaine lettuce", "feta cheese crumbles", "..."]}
```

20 cuisine labels, no missing fields, no empty ingredient lists, unique ids.
