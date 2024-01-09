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

#### Examples
```bash
nohup parallel 'python3 experiment_dbpedia_14.py --group_id dbpedia_14_20240109 --n_sample_from 5 --n_sample_to 25 --n_trials 10 --verification {1} --strategy {2}' ::: themselves dataset ::: normal super > results/ex_20240109.log &
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
    "start_time": {
      "type": "number"
    },
    "finish_time": {
      "type": "number"
    },
    "experiment_id": {
      "type": "string"
    },
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
          "positive_examples": {
            "elements": {
              "type": "string"
            }
          },
          "negative_examples": {
            "elements": {
              "type": "string"
            }
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
