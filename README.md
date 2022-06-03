# PatWay-Net: An interpretable recurrent deep neural network for patient pathway prediction

# Good HPO Settings
- Simulation
  - Normal: lr=0.001, patience=50, batch_size=32, epochs=1000, hidden_per_seq_feat_sz=16, mlp_hidden_size=16, interactions_seq_auto=False, only_static=False
  - Test: lr=0.001, patience=50, batch_size=32, epochs=1000, mlp_hidden_size=8, interactions_seq_auto=False, only_static=True (only static features in data, label age and gender)
  - Test2: lr=0.001, patience=50, batch_size=32, epochs=1000, mlp_hidden_size=8, interactions_seq_auto=False, only_static=True (only gender feature in data and label) 
  - Test3: (all seq feature in data + only stat feature gender in data, only seq. feature crp in label)
  - Test4: (all features in data + only stat features age + gender, only seq. feature crp in label)
- Sepsis


# Data sets
- Sepsis (https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639)




