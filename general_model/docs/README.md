### Some Commonly Used Commands:

#### To Preprocess the Original Dataset:
```
python data/preprocess_original_in_house_dataset.py --mode automatic --partitioning in_house_data_partitioning.yaml
```

#### To Visualiza the Model Architecture:
```
python modeling/model_architectures.py --domain time --task waveform --data in-house
```

#### To Train and Validate:
```
python modeling/training_and_validation.py --domain time --task waveform --data in-house --name <your-experiment-name>
```

#### To Visualize Training and Validation Statistics:
```
python -m tensorboard.main --logdir results/<your-experiment-name>/training_and_validation_statistics
```

#### To Launch Interactive Prediction Plots:
```
python modeling\load_model_and_predict.py --domain time --task waveform --data in-house --name <your-experiment-name>
```
