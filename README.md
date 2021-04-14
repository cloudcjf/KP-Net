##### KP-Net

### Generate pickle files
```
cd generating_queries/

# For training tuples in our baseline network
python generate_training_tuples_baseline.py

# For training tuples in our refined network
python generate_training_tuples_refine.py

# For evaluation tuples
python generate_test_sets.py