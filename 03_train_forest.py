import pandas as pd
import numpy as np
import pickle
from random_forest import RandomForest

def get_mass_bucket(mass):
    # bucket 2 = High Mass Dark Remnant (>3)
    # bucket 1 = Intermediate (1.4 - 3.0)
    # bucket 0 = Soup (<1.4)
    if mass > 3.0:
        return 2
    elif mass >= 1.4:
        return 1
    else:
        return 0
    
def print_confusion_matrix(y_true, y_pred):
    matrix = [[0, 0, 0], 
              [0, 0, 0], 
              [0, 0, 0]]
    
    for true_class, pred_class in zip(y_true, y_pred):
        matrix[true_class][pred_class] += 1
        
    print("\n" + "="*45)
    print("           CONFUSION MATRIX")
    print("="*45)
    print("                 | Predicted Class")
    print("  True Class     |   0      1      2")
    print("-" * 45)
    print(f" 0: Soup (<1.4)  | {matrix[0][0]:<6} {matrix[0][1]:<6} {matrix[0][2]:<6}")
    print(f" 1: Inter (1.4-3)| {matrix[1][0]:<6} {matrix[1][1]:<6} {matrix[1][2]:<6}")
    print(f" 2: HMDR (>3.0)  | {matrix[2][0]:<6} {matrix[2][1]:<6} {matrix[2][2]:<6}")
    print("=" * 45)

if __name__ == "__main__":
    print("loading data...")
    
    features = ["period", "eccentricity", "semi_amplitude_primary", "m1_solar_m", "i_sample_deg"]
    

    cols = features + ["m2_solar", "source_id"] 

    try:
        df_real = pd.read_csv("data/03_physics_anchored/sb1_mc_expanded.csv", usecols=cols)
        df_synth = pd.read_csv("data/03_physics_anchored/synthetic_bh_mc_expanded.csv", usecols=cols)
    except FileNotFoundError:
        print("Error: Couldn't find the csv files")
        exit()

    # combine both datasets together
    df = pd.concat([df_real, df_synth], ignore_index=True)
    
    print("applying target shift...")
    df["target"] = df["m2_solar"].apply(get_mass_bucket)

    print(f"total rows loaded: {len(df)}")
    
    #train/test split by system
    unique_systems = df["source_id"].unique()
    
    # shuffle
    np.random.seed(42) 
    np.random.shuffle(unique_systems)
    
    split_idx = int(len(unique_systems) * 0.85) # 85% for training, 15% for holdout
    
    train_ids = unique_systems[:split_idx]
    test_ids = unique_systems[split_idx:]

    # filter based on where the source id is
    df_train = df[df["source_id"].isin(train_ids)]
    df_test = df[df["source_id"].isin(test_ids)]

    X_train = df_train[features].values
    y_train = df_train["target"].values
    
    X_test = df_test[features].values
    y_test = df_test["target"].values

    print(f"training on {len(X_train)} rows...")
    print(f"testing on {len(X_test)} holdout rows...")

    # train 
    print("\nstarting training phase...")
    rf = RandomForest(n_trees=50, max_depth=10, n_features=4, min_samples_split=50)
    rf.fit(X_train, y_train)
    print("training complete!")

    # see how it did on the holdout set
    print("\nevaluating model...")
    probas = rf.predict_proba(X_test)
    
    # get class with  highest probability
    preds = []
    for p in probas:
        preds.append(np.argmax(p))
    preds = np.array(preds)
    
    accuracy = np.mean(preds == y_test)
    print(f"accuracy on test set: {accuracy * 100:.2f}%")

    print_confusion_matrix(y_test, preds)

    # save model
    print("\nsaving model...")
    with open("data/03_physics_anchored/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
        
    print("saved!")