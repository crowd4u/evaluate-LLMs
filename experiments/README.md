## Experiment with DBpedia 14 dataset

## how to run
### API key
You need to get an API key of OpenAI and set it to `OPENAI_API_KEY` in `.env` file in the same directory as this README.md, like this:
```
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### Run
change directory to `experiments` and run `experiment_dbpedia_14.py` like this:
```bash
cd experiments
python experiment_dbpedia_14.py
```

### Options
You can change the following options in `experiment_dbpedia_14.py`:

## Results
The results are saved in `results/{group_id}` directory.

### Fromat of the result files
The result data in `.piclke` file.
```json
{
    "properties": {
      "args": {
        "type": "object"
      },
      "result": {
"elements": {
  "type": "object",
  "properties": {
    "class label": {
      "type": "string"
    },
    "TP": {
      "type": "number"
    },
    "FP": {
      "type": "number"
    },
    "FN": {
      "type": "number"
    },
    "TN": {
      "type": "number"
    },
    "precision": {
      "type": "number"
    },
    "accuracy": {
      "type": "number"
    },
    "n_samples": {
      "type": "number"
    }
  }
}
      }
    }
}
```
