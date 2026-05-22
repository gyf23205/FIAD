import os, glob
import pandas as pd
import numpy as np

# set the Kaggle dataset path
data_dir = "data/Jet"
print("data_dir:", data_dir)
assert os.path.exists(data_dir), f"data_dir not found: {data_dir}"

# common column names used by C-MAPSS
COL_NAMES = (['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] +
             [f'sensor_{i+1}' for i in range(21)])

def discover_files(data_dir):
    """
    Return dict mapping FD ids ('FD001',...) to dicts with keys possibly 'train','test','rul'.
    This version is tolerant to filename variations and case.
    """
    mapping = {}
    for path in glob.glob(os.path.join(data_dir, "*")):
        name = os.path.basename(path).upper()
        # detect FD id present in filename
        for fd in ['FD001','FD002','FD003','FD004']:
            if fd in name:
                if fd not in mapping:
                    mapping[fd] = {}
                if 'TRAIN' in name:
                    mapping[fd]['train'] = path
                if 'TEST' in name:
                    mapping[fd]['test'] = path
                if 'RUL' in name or name.startswith('RUL_') or 'RUL' in name:
                    mapping[fd]['rul'] = path
                # also accept files named like 'train_FD001.txt' etc.
                # note: we allow multiple matches; last one wins (fine for typical datasets)
    return mapping

def load_cmapss_pair(train_path, test_path, rul_path=None):
    """Load a single FD dataset trio and compute per-row RUL for train and test (if RUL provided)."""
    train = pd.read_csv(train_path, sep='\s+', header=None, names=COL_NAMES)
    train['RUL'] = train.groupby('unit')['cycle'].transform('max') - train['cycle']

    test = pd.read_csv(test_path, sep='\s+', header=None, names=COL_NAMES)
    if rul_path is not None and os.path.exists(rul_path):
        rul = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])
        max_cycle = test.groupby('unit')['cycle'].max().reset_index().sort_values('unit').reset_index(drop=True)
        max_cycle = max_cycle.sort_values('unit').reset_index(drop=True)
        if len(rul) != len(max_cycle):
            print("Warning: RUL file length != number of test units. Skipping per-row test RUL.")
            test['RUL'] = np.nan
        else:
            max_cycle['RUL_final'] = rul['RUL'].values
            test = test.merge(max_cycle[['unit','RUL_final']], on='unit', how='left')
            test['RUL'] = test.groupby('unit')['cycle'].transform('max') - test['cycle'] + test['RUL_final']
            test = test.drop(columns=['RUL_final'])
    else:
        test['RUL'] = np.nan

    train[['unit','cycle']] = train[['unit','cycle']].astype(int)
    test[['unit','cycle']] = test[['unit','cycle']].astype(int)
    return train.reset_index(drop=True), test.reset_index(drop=True)

# discover dataset files
mapping = discover_files(data_dir)
print("Discovered dataset keys and file types:")
for k,v in mapping.items():
    print(k, v.keys())

# --- Choose dataset FDID to run (change to FD002/FD003/FD004 as needed) ---
FDID = 'FD001'   # <------ change here if you want FD002/FD003/FD004

# If chosen FDID doesn't have train+test, pick the first available FD that has both
if FDID not in mapping or 'train' not in mapping[FDID] or 'test' not in mapping[FDID]:
    print(f"Requested {FDID} is missing train/test. Searching for first FD with both train and test...")
    chosen = None
    for fd, files in mapping.items():
        if 'train' in files and 'test' in files:
            chosen = fd
            break
    if chosen is None:
        raise FileNotFoundError(f"No FD dataset with both train and test found in {data_dir}. Mapping: {mapping}")
    print(f"Switching to available dataset: {chosen}")
    FDID = chosen

train_path = mapping[FDID]['train']
test_path  = mapping[FDID]['test']
rul_path   = mapping[FDID].get('rul', None)

print("Using:", train_path, test_path, rul_path)
train_df, test_df = load_cmapss_pair(train_path, test_path, rul_path)
print(f"Loaded {FDID}: train rows={len(train_df)} (units={train_df['unit'].nunique()}), test rows={len(test_df)} (units={test_df['unit'].nunique()})")


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# reproducibility
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DataPreprocessor:
    def __init__(self):
        # Use MinMax by default — safe when features have different ranges
        self.scaler = MinMaxScaler()
        self.feature_cols = None

    def fit_scaler(self, df):
        """Fit scaler on training data features"""
        self.feature_cols = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
        self.scaler.fit(df[self.feature_cols].values)

    def preprocess(self, df, fit=False):
        """Return a deep-copied scaled dataframe. If fit=True, fit scaler from df."""
        if fit or self.feature_cols is None:
            self.fit_scaler(df)
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols].values)
        return df_scaled

    def create_sequences(self, df, sequence_length=30, stride=1):
        """
        Create sliding windows per unit for sequence models.
        Returns X: (n_windows, seq_len, n_features), y: (n_windows,)
        """
        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
        sequences = []
        targets = []
        # ensure sorted by unit and cycle
        for uid in sorted(df['unit'].unique()):
            unit_df = df[df['unit'] == uid].sort_values('cycle')
            X_unit = unit_df[self.feature_cols].values
            y_unit = unit_df['RUL'].values
            L = len(unit_df)
            if L <= sequence_length:
                continue
            for start in range(0, L - sequence_length, stride):
                end = start + sequence_length
                sequences.append(X_unit[start:end])
                targets.append(y_unit[end])  # predict RUL at next step after window (align with earlier code)
        if len(sequences) == 0:
            return np.empty((0, sequence_length, len(self.feature_cols))), np.empty((0,))
        return np.array(sequences), np.array(targets)

# Instantiate and run preprocessing
preprocessor = DataPreprocessor()
train_df_scaled = preprocessor.preprocess(train_df, fit=True)
test_df_scaled  = preprocessor.preprocess(test_df, fit=False)

feature_cols = preprocessor.feature_cols
print("Feature columns:", feature_cols)
print("Train shape:", train_df_scaled.shape, "Test shape:", test_df_scaled.shape)

# Flat ML features
X_train_flat = train_df_scaled[feature_cols].values
y_train_flat = train_df_scaled['RUL'].values
X_test_flat  = test_df_scaled[feature_cols].values
y_test_flat  = test_df_scaled['RUL'].values

print("Flat feature shapes -> X_train:", X_train_flat.shape, "X_test:", X_test_flat.shape)

# Sequence creation: quick debug option to limit number of units
# For fast Kaggle debugging set n_units_limit to small int, else set to None
n_units_limit = None   # e.g., 40 for quick runs
if n_units_limit is not None:
    train_small = train_df_scaled[train_df_scaled['unit'] <= n_units_limit]
    test_small  = test_df_scaled[test_df_scaled['unit'] <= max(1, n_units_limit//2)]
else:
    train_small = train_df_scaled
    test_small  = test_df_scaled

SEQ_LEN = 100
X_train_seq, y_train_seq = preprocessor.create_sequences(train_small, sequence_length=SEQ_LEN, stride=1)
X_test_seq,  y_test_seq  = preprocessor.create_sequences(test_small,  sequence_length=SEQ_LEN, stride=1)
print("Sequence shapes -> X_train_seq:", X_train_seq.shape, "y_train_seq:", y_train_seq.shape)
print("                 X_test_seq: ", X_test_seq.shape, "y_test_seq:", y_test_seq.shape)
print("unique RUL values in train:", np.unique(y_train_seq))
print("unique RUL values in test:", np.unique(y_test_seq))

# Save the processed data to npy files
# np.save("X_train_flat.npy", X_train_flat)
# np.save("y_train_flat.npy", y_train_flat)
# np.save("X_test_flat.npy", X_test_flat)
# np.save("y_test_flat.npy", y_test_flat)
# np.save("data/Jet/X_train_seq.npy", X_train_seq)
# np.save("data/Jet/y_train_seq.npy", y_train_seq)
# np.save("data/Jet/X_test_seq.npy", X_test_seq)
# np.save("data/Jet/y_test_seq.npy", y_test_seq)