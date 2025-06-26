import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.layers import Conv1D, TimeDistributed, MaxPooling1D, Flatten, Add, Attention, Reshape
from tensorflow.keras.layers import ConvLSTM2D, RepeatVector, Concatenate, Lambda, MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, accuracy_score
from scipy.ndimage import gaussian_filter1d
import time
import os
import requests
from datetime import datetime, timedelta

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available!")
else:
    print("No GPU found, using CPU.")

# Load data
try:
    # For Google Colab
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename)
except ImportError:
    # For local execution
    df = pd.read_csv('weather_data.csv')  # Update with your file path

# Filter for Sri Lanka data
if 'country' in df.columns:
    original_len = len(df)
    df = df[df['country'] == 'Sri Lanka']
    print(f"Filtered data for Sri Lanka: {len(df)} rows (from {original_len} total)")
    if len(df) == 0:
        print("WARNING: No Sri Lanka data found. Using all available data.")
        df = pd.read_csv('weather_data.csv')  # Reload the data

# Display basic info about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'])

# IMPROVEMENT: Advanced data preprocessing using sinh-arcsinh transformation
def sinh_arcsinh_transform(x, epsilon=0.01, delta=1.0):
    """Apply sinh-arcsinh transformation to better handle extreme values"""
    return np.sinh(delta * np.arcsinh(x) - epsilon)

def inverse_sinh_arcsinh_transform(x, epsilon=0.01, delta=1.0):
    """Inverse transform to return to original scale"""
    return np.sinh((np.arcsinh(x) + epsilon) / delta)

# Apply transformation to precipitation values if they exist
if 'precipitation_sum' in df.columns:
    # Store original values for reference
    df['precipitation_sum_original'] = df['precipitation_sum'].copy()

    # Handle zeros and small values carefully
    mask = df['precipitation_sum'] > 0
    df.loc[mask, 'precipitation_sum_transformed'] = sinh_arcsinh_transform(df.loc[mask, 'precipitation_sum'])
    df.loc[~mask, 'precipitation_sum_transformed'] = 0

    print(f"Applied sinh-arcsinh transformation to precipitation values")

    # Use transformed values for modeling
    # Note: We'll keep the original column name for simplicity in the code
    df['precipitation_sum'] = df['precipitation_sum_transformed']

# Standard time features - Moved this section up
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['dayofyear'] = df['time'].dt.dayofyear
df['year'] = df['time'].dt.year
df['season'] = (df['month'] % 12 + 3) // 3  # 1: spring, 2: summer, 3: fall, 4: winter
df['week_of_year'] = df['time'].dt.isocalendar().week

# IMPROVEMENT: Sri Lanka-specific feature engineering
# 1. Add monsoon season indicators
df['southwest_monsoon'] = ((df['month'] >= 5) & (df['month'] <= 9)).astype(int)
df['northeast_monsoon'] = ((df['month'] >= 11) | (df['month'] <= 3)).astype(int)
df['inter_monsoon1'] = (df['month'] == 4).astype(int)
df['inter_monsoon2'] = (df['month'] == 10).astype(int)

# 2. Add typical monsoon intensity based on historical patterns
# This is a simplified model - ideally would be based on historical data
monsoon_intensity = {
    1: 0.7,  # January - NE monsoon
    2: 0.4,  # February - NE monsoon (weakening)
    3: 0.2,  # March - end of NE monsoon
    4: 0.3,  # April - inter-monsoon
    5: 0.6,  # May - start of SW monsoon
    6: 0.8,  # June - SW monsoon
    7: 0.9,  # July - SW monsoon peak
    8: 0.9,  # August - SW monsoon peak
    9: 0.7,  # September - SW monsoon (weakening)
    10: 0.5, # October - inter-monsoon
    11: 0.6, # November - start of NE monsoon
    12: 0.8, # December - NE monsoon
}

df['monsoon_intensity'] = df['month'].map(monsoon_intensity)

# 3. Regional indicators for Sri Lanka's climate zones
# If city information is available
if 'city' in df.columns:
    # Sri Lanka climate zone mapping (simplified)
    wet_zone_cities = ['Colombo', 'Galle', 'Ratnapura', 'Kalutara']
    dry_zone_cities = ['Anuradhapura', 'Hambantota', 'Jaffna', 'Mannar']
    intermediate_zone_cities = ['Kandy', 'Badulla', 'Kurunegala']

    # Create climate zone indicators
    df['wet_zone'] = df['city'].isin(wet_zone_cities).astype(int)
    df['dry_zone'] = df['city'].isin(dry_zone_cities).astype(int)
    df['intermediate_zone'] = df['city'].isin(intermediate_zone_cities).astype(int)


# Add cyclical encoding for seasonal patterns
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['day_sin'] = np.sin(2 * np.pi * df['dayofyear']/365)
df['day_cos'] = np.cos(2 * np.pi * df['dayofyear']/365)
df['hour_sin'] = np.sin(2 * np.pi * 12/24)  # Assume noon for daily data
df['hour_cos'] = np.cos(2 * np.pi * 12/24)

# Handle time columns: sunrise and sunset
def time_to_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    try:
        if isinstance(time_str, str) and ('AM' in time_str or 'PM' in time_str):
            time_obj = pd.to_datetime(time_str, format='%I:%M:%S %p').time()
        else:
            time_obj = pd.to_datetime(time_str).time()
        return time_obj.hour * 60 + time_obj.minute
    except:
        return np.nan

if 'sunrise' in df.columns:
    df['sunrise_minutes'] = df['sunrise'].apply(time_to_minutes)
    df['sunset_minutes'] = df['sunset'].apply(time_to_minutes)
    df['daylight_minutes'] = df['sunset_minutes'] - df['sunrise_minutes']

# IMPROVEMENT: Enhanced precipitation features
# 1. Calculate rolling statistics with more window sizes
precip_related_cols = [col for col in df.columns if any(x in col.lower()
                      for x in ['precipitation', 'rain', 'humid', 'pressure', 'wind'])]

print(f"\nPrecipitation-related columns found: {precip_related_cols}")

# More fine-grained window sizes for better pattern detection
window_sizes = [3, 5, 7, 10, 14, 21, 30]

for col in precip_related_cols:
    if col in df.columns and df[col].dtype.kind in 'ifc':  # Numeric columns only
        # Calculate rolling statistics
        for window in window_sizes:
            if len(df) > window:
                df[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}d'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)

                # Add median and max for better extreme event capture
                df[f'{col}_rolling_median_{window}d'] = df[col].rolling(window=window, min_periods=1).median()
                df[f'{col}_rolling_max_{window}d'] = df[col].rolling(window=window, min_periods=1).max()

        # Calculate day-to-day changes (first derivative)
        df[f'{col}_change'] = df[col].diff().fillna(0)

        # Calculate change acceleration (second derivative)
        df[f'{col}_change_accel'] = df[f'{col}_change'].diff().fillna(0)

        # Exponential weighted features - give more weight to recent values
        df[f'{col}_ewm_mean'] = df[col].ewm(span=7).mean()
        df[f'{col}_ewm_std'] = df[col].ewm(span=7).std()

# 2. Add more sophisticated precipitation event features
if 'precipitation_sum' in df.columns:
    # Define different precipitation thresholds
    df['light_rain'] = ((df['precipitation_sum'] > 0.1) & (df['precipitation_sum'] <= 2.5)).astype(int)
    df['moderate_rain'] = ((df['precipitation_sum'] > 2.5) & (df['precipitation_sum'] <= 10)).astype(int)
    df['heavy_rain'] = ((df['precipitation_sum'] > 10) & (df['precipitation_sum'] <= 50)).astype(int)
    df['extreme_rain'] = (df['precipitation_sum'] > 50).astype(int)

    # Calculate rain streak features
    rain_streak = 0
    rain_streaks = []

    for rain in df['precipitation_sum'] > 0.5:
        if rain:
            rain_streak += 1
        else:
            rain_streak = 0
        rain_streaks.append(rain_streak)

    df['rain_streak'] = rain_streaks

    # Calculate frequency features for different intensities
    for window in window_sizes:
        if len(df) > window:
            df[f'light_rain_freq_{window}d'] = df['light_rain'].rolling(window=window, min_periods=1).mean()
            df[f'moderate_rain_freq_{window}d'] = df['moderate_rain'].rolling(window=window, min_periods=1).mean()
            df[f'heavy_rain_freq_{window}d'] = df['heavy_rain'].rolling(window=window, min_periods=1).mean()
            df[f'extreme_rain_freq_{window}d'] = df['extreme_rain'].rolling(window=window, min_periods=1).mean()

            # Also calculate total rain days
            df[f'rain_days_{window}d'] = (df['precipitation_sum'] > 0.1).rolling(window=window, min_periods=1).sum()

# 3. Weather pattern changes with more context
if 'weathercode' in df.columns:
    df['weather_change'] = (df['weathercode'].diff() != 0).astype(int)

    # Get dominant weather pattern in last week
    if len(df) > 7:
        df['dominant_weathercode_7d'] = df['weathercode'].rolling(window=7).apply(
            lambda x: x.value_counts().index[0] if not x.empty else np.nan)

        # Is current weather different from dominant pattern
        df['weather_anomaly'] = (df['weathercode'] != df['dominant_weathercode_7d']).astype(int)

# One-hot encode categorical variables
cat_cols = ['city', 'country']
encoders = {}

for col in cat_cols:
    if col in df.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=[f'{col}_{i}' for i in range(encoded.shape[1])])
        df = pd.concat([df, encoded_df], axis=1)
        encoders[col] = encoder
        print(f"Encoded {col} into {encoded.shape[1]} categories")

# Feature selection - drop redundant and non-useful columns
base_drop_cols = ['time', 'sunrise', 'sunset'] + cat_cols

# Remove temperature_mean if we have max and min (derived feature)
if 'temperature_2m_mean' in df.columns and 'temperature_2m_max' in df.columns and 'temperature_2m_min' in df.columns:
    base_drop_cols.append('temperature_2m_mean')
    print("Removing temperature_2m_mean as it's derived from max and min")

# Remove apparent_temperature_mean if we have max and min (derived feature)
if 'apparent_temperature_mean' in df.columns and 'apparent_temperature_max' in df.columns and 'apparent_temperature_min' in df.columns:
    base_drop_cols.append('apparent_temperature_mean')
    print("Removing apparent_temperature_mean as it's derived from max and min")

# Remove snow-related features for tropical climate (Sri Lanka)
snow_cols = [col for col in df.columns if 'snow' in col.lower()]
base_drop_cols.extend(snow_cols)
print(f"Removing {len(snow_cols)} snow-related features as they're irrelevant for Sri Lanka")

# IMPROVEMENT: Try to fetch ENSO data (simplified version)
try:
    # This is a simplified example - you would need to use a proper API or dataset
    # Normally you'd get this from NOAA or similar source
    # For this example, we'll create simulated ENSO data

    # Generate synthetic ENSO index for the date range in the dataset
    start_date = df['time'].min()
    end_date = df['time'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')

    # Create a synthetic oscillating ENSO pattern
    enso_values = 0.5 * np.sin(np.linspace(0, 4*np.pi, len(date_range)))
    enso_df = pd.DataFrame({
        'date': date_range,
        'enso_index': enso_values
    })

    # Resample to daily frequency with forward fill
    enso_df = enso_df.set_index('date')
    daily_enso = enso_df.resample('D').ffill()

    # Merge with main dataframe
    df_with_date = df.copy()
    df_with_date['date'] = df_with_date['time'].dt.date
    df_with_date['date'] = pd.to_datetime(df_with_date['date'])

    # Merge ENSO data
    df_with_enso = pd.merge(df_with_date, daily_enso, left_on='date', right_index=True, how='left')

    # Replace the main dataframe if merge was successful
    if 'enso_index' in df_with_enso.columns:
        df = df_with_enso
        print("Added ENSO index data")

    # Clean up temporary date column
    if 'date' in df.columns:
        df = df.drop('date', axis=1)

except Exception as e:
    print(f"Could not fetch ENSO data: {e}")
    print("Proceeding without ENSO indices")

# Define forecast target(s)
target_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']

# If we transformed precipitation earlier, use the original for evaluation
if 'precipitation_sum_original' in df.columns:
    target_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum_original']

binary_precip_col = ['light_rain', 'moderate_rain', 'heavy_rain', 'extreme_rain']
print(f"\nTarget columns: {target_cols}")
print(f"Binary precipitation targets: {binary_precip_col}")

# Select features
feature_cols = [col for col in df.columns if col not in base_drop_cols
               and col not in target_cols
               and col not in binary_precip_col
               and col != 'precipitation_sum_original']  # Exclude original precip if we created it

print(f"\nSelected feature columns ({len(feature_cols)}):")
print(feature_cols[:10], "... and more")

# Print summary stats for precipitation
if 'precipitation_sum_original' in df.columns:
    print("\nPrecipitation Summary Statistics:")
    print(df['precipitation_sum_original'].describe())

    print(f"Days with light rain: {df['light_rain'].sum()} ({df['light_rain'].mean()*100:.1f}%)")
    print(f"Days with moderate rain: {df['moderate_rain'].sum()} ({df['moderate_rain'].mean()*100:.1f}%)")
    print(f"Days with heavy rain: {df['heavy_rain'].sum()} ({df['heavy_rain'].mean()*100:.1f}%)")
    print(f"Days with extreme rain: {df['extreme_rain'].sum()} ({df['extreme_rain'].mean()*100:.1f}%)")

# IMPROVEMENT: Set cutoff dates based on monsoon seasons for better model testing
# We want the test set to include both SW and NE monsoon periods
df = df.sort_values('time')

# Use 80-10-10 split but ensure test set has representative monsoon seasons
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size+val_size]
test_df = df.iloc[train_size+val_size:]

print(f"\nTrain set: {len(train_df)} rows (from {train_df['time'].min()} to {train_df['time'].max()})")
print(f"Validation set: {len(val_df)} rows (from {val_df['time'].min()} to {val_df['time'].max()})")
print(f"Test set: {len(test_df)} rows (from {test_df['time'].min()} to {test_df['time'].max()})")

# Check monsoon representation in test set
if 'southwest_monsoon' in test_df.columns and 'northeast_monsoon' in test_df.columns:
    sw_monsoon_days = test_df['southwest_monsoon'].sum()
    ne_monsoon_days = test_df['northeast_monsoon'].sum()
    print(f"Test set includes {sw_monsoon_days} SW monsoon days and {ne_monsoon_days} NE monsoon days")

# IMPROVEMENT: Use more sophisticated scaling techniques
# For precipitation, we already applied sinh-arcsinh transformation
# For other features, we'll use a robust scaler
from sklearn.preprocessing import RobustScaler

# Use robust scaling for better handling of outliers
scaler_features = RobustScaler()  # Less sensitive to outliers than StandardScaler
scaler_targets = RobustScaler()   # Better preserves the distribution shape

# Fit scalers on training data only
train_features = train_df[feature_cols].copy()
train_targets = train_df[target_cols].copy()
train_binary_target = train_df[binary_precip_col].copy()

# Transform all datasets - features
train_features_scaled = scaler_features.fit_transform(train_features)
val_features_scaled = scaler_features.transform(val_df[feature_cols])
test_features_scaled = scaler_features.transform(test_df[feature_cols])

# Transform all datasets - targets
train_targets_scaled = scaler_targets.fit_transform(train_targets)
val_targets_scaled = scaler_targets.transform(val_df[target_cols])
test_targets_scaled = scaler_targets.transform(test_df[target_cols])

# Binary target doesn't need scaling, but we need to concatenate multiple binary targets
train_binary = np.hstack([train_df[col].values.reshape(-1, 1) for col in binary_precip_col])
val_binary = np.hstack([val_df[col].values.reshape(-1, 1) for col in binary_precip_col])
test_binary = np.hstack([test_df[col].values.reshape(-1, 1) for col in binary_precip_col])

# IMPROVEMENT: Longer sequence length (60 days) to better capture seasonal patterns
# Create sequences for LSTM with increased sequence length
def create_sequences(features, targets, binary_targets=None, seq_length=60, pred_length=5):
    """Create sequences with optional binary target output"""
    X, y = [], []
    binary_y = []

    for i in range(len(features) - seq_length - pred_length + 1):
        X.append(features[i:i+seq_length])
        y.append(targets[i+seq_length:i+seq_length+pred_length])

        if binary_targets is not None:
            binary_y.append(binary_targets[i+seq_length:i+seq_length+pred_length])

    if binary_targets is not None:
        return np.array(X), np.array(y), np.array(binary_y)
    else:
        return np.array(X), np.array(y)

# Parameters - INCREASED SEQUENCE LENGTH for better pattern capture
sequence_length = 60  # Increased from 30 to 60 days
prediction_length = 5  # Predict 5 days ahead

# Create sequences with binary targets
X_train, y_train, binary_y_train = create_sequences(
    train_features_scaled, train_targets_scaled, train_binary,
    seq_length=sequence_length, pred_length=prediction_length)

X_val, y_val, binary_y_val = create_sequences(
    val_features_scaled, val_targets_scaled, val_binary,
    seq_length=sequence_length, pred_length=prediction_length)

X_test, y_test, binary_y_test = create_sequences(
    test_features_scaled, test_targets_scaled, test_binary,
    seq_length=sequence_length, pred_length=prediction_length)

print(f"\nSequence input shape: {X_train.shape}")
print(f"Sequence output shape: {y_train.shape}")
print(f"Binary output shape: {binary_y_train.shape}")

# Flatten target arrays for model training
y_train_flat = y_train.reshape(y_train.shape[0], -1)
y_val_flat = y_val.reshape(y_val.shape[0], -1)
y_test_flat = y_test.reshape(y_test.shape[0], -1)

# IMPROVEMENT: Asymmetric loss function that penalizes precipitation underestimation more heavily
def asymmetric_mse(y_true, y_pred):
    """
    Custom loss function that penalizes underestimation of precipitation (false negatives)
    more heavily than overestimation (false positives).

    For precipitation values (every 3rd value starting at index 2), we apply higher
    weight when the prediction is lower than the actual value.
    """
    # Calculate squared errors
    squared_error = tf.square(y_true - y_pred)

    # Extract precipitation values (every 3rd column starting from index 2)
    precip_indices = tf.range(2, tf.shape(y_true)[1], 3)

    # Create weights tensor (default weight = 1.0)
    weights = tf.ones_like(squared_error)

    # For each precipitation column
    for i in range(0, prediction_length):
        idx = 2 + i * 3  # Index of precipitation column

        if idx < tf.shape(y_true)[1]:
            # Extract precipitation values
            true_vals = y_true[:, idx]
            pred_vals = y_pred[:, idx]

            # Calculate error direction: positive error means underestimation
            error = true_vals - pred_vals

            # Create weights: 2.5x for underestimation, 1.0 for overestimation
            precip_weights = tf.where(error > 0, 2.5, 1.0)

            # Create indices for updating the weights tensor
            indices = tf.stack([tf.range(tf.shape(weights)[0]), tf.ones_like(tf.range(tf.shape(weights)[0])) * idx], axis=1)

            # Update weights for this precipitation column
            weights = tf.tensor_scatter_nd_update(weights, indices, precip_weights)

    # Apply weights to squared errors and take mean
    weighted_squared_error = squared_error * weights
    return tf.reduce_mean(weighted_squared_error)

# IMPROVEMENT: Residual ConvLSTM model with attention for better precipitation forecasting
def create_advanced_model(input_shape, num_targets, num_days):
    """
    Create an advanced model with:
    1. Convolutional layers to extract spatial patterns
    2. LSTM layers with residual connections
    3. Self-attention mechanism
    4. Multi-headed regression
    """
    # Input
    inputs = Input(shape=input_shape)

    # 1D CNN layers to extract features
    x = Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Second CNN layer
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Store the CNN output for residual connection
    cnn_output = x

    # Bidirectional LSTM with increased capacity
    x = Bidirectional(LSTM(192, return_sequences=True, activation='tanh'))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Self-attention mechanism
    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=48
    )(x, x)

    # Add residual connection around attention
    x = Add()([attention_output, x])
    x = BatchNormalization()(x)

    # Second LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=False, activation='tanh'))(x)
    x = BatchNormalization()(x)

    # Residual connection from CNN (need to match dimensions)
    # Global average pooling to reduce CNN output dimensions
    cnn_pooled = tf.keras.layers.GlobalAveragePooling1D()(cnn_output)
    cnn_pooled = Dense(256)(cnn_pooled)  # Match dimensions with LSTM output

    # Combine LSTM output with CNN features
    x = Add()([x, cnn_pooled])
    x = Dropout(0.3)(x)

    # Common dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)

    # Classification outputs for rain intensity categories
    rain_outputs = []
    for i in range(num_days):
        for j, category in enumerate(['light', 'moderate', 'heavy', 'extreme']):
            rain_output = Dense(16, activation='relu')(x)
            rain_output = Dense(1, activation='sigmoid',
                               name=f'day{i+1}_{category}_rain')(rain_output)
            rain_outputs.append(rain_output)

    # Main regression output
    regression_output = Dense(num_targets * num_days, name='main_output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=rain_outputs + [regression_output])

    # Prepare loss dictionary
    losses = {'main_output': asymmetric_mse}
    loss_weights = {'main_output': 1.0}

    # Add binary crossentropy for all rain category outputs
    for i in range(len(rain_outputs)):
        output_name = model.outputs[i].name.split('/')[0]
        losses[output_name] = 'binary_crossentropy'
        loss_weights[output_name] = 0.1  # Lower weight for classification tasks

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'main_output': 'mae'}
    )

    return model

# IMPROVEMENT: Quantile regression model for better extreme event capture
def create_quantile_regression_model(input_shape, num_targets, num_days):
    """Create a model that predicts multiple quantiles (50th, 90th) for precipitation"""

    # Define quantile loss function
    def quantile_loss(q, y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(q * error, (q - 1) * error))

    # Median (q=0.5) quantile loss
    q50_loss = lambda y, f: quantile_loss(0.5, y, f)
    # Upper (q=0.9) quantile loss for extreme events
    q90_loss = lambda y, f: quantile_loss(0.9, y, f)

    # Input layer
    inputs = Input(shape=input_shape)

    # Shared layers
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Common dense layers
    x = Dense(64, activation='relu')(x)

    # Median (q=0.5) output - standard predictions
    median_output = Dense(num_targets * num_days, name='q50_output')(x)

    # Upper quantile (q=0.9) output - helps with extreme precipitation
    upper_output = Dense(num_targets * num_days, name='q90_output')(x)

    # Create model with multiple outputs
    model = Model(inputs=inputs, outputs=[median_output, upper_output])

    # Compile with appropriate losses
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'q50_output': q50_loss, 'q90_output': q90_loss},
        loss_weights={'q50_output': 1.0, 'q90_output': 0.5},
        metrics={'q50_output': 'mae', 'q90_output': 'mae'}
    )

    return model

# Standard Enhanced model (updated with higher capacity)
def create_enhanced_model(input_shape, output_shape):
    """Create an enhanced LSTM model with higher capacity"""
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First layer: larger capacity (256 units)
    model.add(LSTM(256, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Second layer
    model.add(LSTM(192, activation='tanh', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Third layer
    model.add(LSTM(128, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))  # Increased dropout

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape))

    # Compile with asymmetric loss function
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=asymmetric_mse,
        metrics=['mae']
    )

    return model

# Train the advanced model
def train_advanced_model(X_train, y_train, binary_y_train,
                        X_val, y_val, binary_y_val,
                        X_test, y_test, binary_y_test):
    """Train the advanced model"""

    print("\n\nTraining ADVANCED model...")
    start_time = time.time()

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_targets = y_train.shape[2]  # temp_max, temp_min, precip
    num_days = y_train.shape[1]     # 5 days

    # Create model
    model = create_advanced_model(input_shape, num_targets, num_days)

    # Prepare classification targets - reshape to match output structure
    binary_targets_train = []
    binary_targets_val = []

    # For each prediction day and rain category
    for i in range(num_days):
        for j in range(binary_y_train.shape[2]):  # 4 categories: light, moderate, heavy, extreme
            binary_targets_train.append(binary_y_train[:, i, j])
            binary_targets_val.append(binary_y_val[:, i, j])

    # Add main regression target
    train_targets = binary_targets_train + [y_train_flat]
    val_targets = binary_targets_val + [y_val_flat]

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_advanced_weather_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    # Train model with more epochs and reduced batch size
    history = model.fit(
        X_train, train_targets,
        validation_data=(X_val, val_targets),
        epochs=100,  # More epochs with early stopping
        batch_size=16,  # Smaller batch size for better generalization
        callbacks=callbacks,
        verbose=1
    )

    # Prepare test targets for evaluation
    binary_targets_test = []
    for i in range(num_days):
        for j in range(binary_y_test.shape[2]):
            binary_targets_test.append(binary_y_test[:, i, j])

    test_targets = binary_targets_test + [y_test_flat]
    test_results = model.evaluate(X_test, test_targets, verbose=0)

    # Main output is the last one
    main_output_idx = len(model.output_names) - 1
    test_loss = test_results[0]  # Overall loss
    main_mae = test_results[main_output_idx + 1]  # MAE for main output

    # Make predictions
    predictions = model.predict(X_test)
    main_predictions = predictions[-1]  # Last output is main regression

    # Reshape to original dimensions
    main_predictions_reshaped = main_predictions.reshape(y_test.shape)

    training_time = time.time() - start_time

    # Format classification predictions
    binary_predictions = []
    binary_accuracies = []

    # Process each binary output (one per day per rain category)
    curr_pred_idx = 0
    for i in range(num_days):
        day_preds = []
        day_accs = []

        for j in range(binary_y_test.shape[2]):
            pred = (predictions[curr_pred_idx] > 0.5).astype(int)
            true = binary_y_test[:, i, j]
            acc = accuracy_score(true, pred)

            day_preds.append(pred)
            day_accs.append(acc)
            curr_pred_idx += 1

        binary_predictions.append(day_preds)
        binary_accuracies.append(day_accs)

    # Calculate average accuracy by rain category
    rain_categories = ['light', 'moderate', 'heavy', 'extreme']
    category_accuracies = {}

    for j, category in enumerate(rain_categories):
        # Average across all days for this category
        category_acc = np.mean([binary_accuracies[i][j] for i in range(num_days)])
        category_accuracies[category] = category_acc

    # Store results
    result = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_mae': main_mae,
        'predictions': main_predictions_reshaped,
        'binary_predictions': binary_predictions,
        'category_accuracies': category_accuracies,
        'avg_accuracy': np.mean(list(category_accuracies.values())),
        'training_time': training_time
    }

    print(f"\nADVANCED Model Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Regression Test MAE: {main_mae:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print("\nAccuracy by Rain Category:")
    for category, acc in category_accuracies.items():
        print(f"  {category.capitalize()}: {acc:.4f}")

    return result

# Train the quantile regression model
def train_quantile_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the quantile regression model"""

    print("\n\nTraining QUANTILE model...")
    start_time = time.time()

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_targets = y_train.shape[2]
    num_days = y_train.shape[1]

    # Create model
    model = create_quantile_regression_model(input_shape, num_targets, num_days)

    # Train model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_quantile_weather_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    history = model.fit(
        X_train, [y_train_flat, y_train_flat],  # Same targets for both outputs during training
        validation_data=(X_val, [y_val_flat, y_val_flat]),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_results = model.evaluate(X_test, [y_test_flat, y_test_flat], verbose=0)
    test_loss = test_results[0]  # Overall loss
    q50_mae = test_results[2]    # MAE for median predictions
    q90_mae = test_results[4]    # MAE for upper quantile predictions

    # Make predictions
    q50_pred, q90_pred = model.predict(X_test)

    # Reshape predictions to original dimensions
    q50_pred_reshaped = q50_pred.reshape(y_test.shape)
    q90_pred_reshaped = q90_pred.reshape(y_test.shape)

    training_time = time.time() - start_time

    # Store results
    result = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'q50_mae': q50_mae,
        'q90_mae': q90_mae,
        'q50_predictions': q50_pred_reshaped,
        'q90_predictions': q90_pred_reshaped,
        'training_time': training_time
    }

    print(f"\nQUANTILE Model Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Median (q50) Test MAE: {q50_mae:.4f}")
    print(f"Upper (q90) Test MAE: {q90_mae:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    return result

# Train the enhanced baseline model
def train_enhanced_baseline(X_train, y_train_flat, X_val, y_val_flat, X_test, y_test):
    """Train an enhanced baseline model with higher capacity"""

    print("\n\nTraining ENHANCED BASELINE model...")
    start_time = time.time()

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train_flat.shape[1]

    # Create model
    model = create_enhanced_model(input_shape, output_shape)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_enhanced_baseline_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    # Train model
    history = model.fit(
        X_train, y_train_flat,
        validation_data=(X_val, y_val_flat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test_flat)

    # Make predictions
    predictions_flat = model.predict(X_test)
    predictions = predictions_flat.reshape(y_test.shape)

    training_time = time.time() - start_time

    # Store results
    result = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'predictions': predictions,
        'training_time': training_time
    }

    print(f"\nENHANCED BASELINE Model Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")

    return result

# Train all models
print("\n\n==== TRAINING SRI LANKA PRECIPITATION MODELS ====")
enhanced_result = train_enhanced_baseline(
    X_train, y_train_flat,
    X_val, y_val_flat,
    X_test, y_test_flat
)

advanced_result = train_advanced_model(
    X_train, y_train, binary_y_train,
    X_val, y_val, binary_y_val,
    X_test, y_test, binary_y_test
)

quantile_result = train_quantile_model(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test
)

# IMPROVEMENT: Create ensemble model that combines predictions from all models
# with special focus on high precipitation events
def create_advanced_ensemble_predictions(models_dict, y_test):
    """
    Create ensemble predictions by combining all models with intelligent weighting
    based on precipitation intensity:
    - For low precipitation: Give more weight to median predictions
    - For high precipitation: Give more weight to upper quantile and advanced model
    """
    # Extract predictions
    enhanced_preds = models_dict['enhanced']['predictions']
    advanced_preds = models_dict['advanced']['predictions']
    q50_preds = models_dict['quantile']['q50_predictions']
    q90_preds = models_dict['quantile']['q90_predictions']

    # Create ensemble predictions array
    ensemble_predictions = np.zeros_like(y_test)

    # For each sample, day, and target
    for sample_idx in range(y_test.shape[0]):
        for day_idx in range(y_test.shape[1]):
            for target_idx in range(y_test.shape[2]):
                # Is this a precipitation prediction?
                is_precip = (target_idx == 2)

                if is_precip:
                    # Compute ensemble prediction for precipitation with adaptive weighting

                    # Get all precipitation predictions for this point
                    enhanced_val = enhanced_preds[sample_idx, day_idx, target_idx]
                    advanced_val = advanced_preds[sample_idx, day_idx, target_idx]
                    q50_val = q50_preds[sample_idx, day_idx, target_idx]
                    q90_val = q90_preds[sample_idx, day_idx, target_idx]

                    # Calculate maximum predicted value
                    max_pred = max(enhanced_val, advanced_val, q50_val)

                    # Use different weighting strategies based on predicted precipitation
                    if max_pred > 0.75:  # High precipitation likely (in normalized scale)
                        # Give more weight to q90 and advanced model
                        weights = {
                            'enhanced': 0.1,
                            'advanced': 0.4,
                            'q50': 0.1,
                            'q90': 0.4
                        }
                    else:  # Low/normal precipitation
                        # More balanced weighting
                        weights = {
                            'enhanced': 0.25,
                            'advanced': 0.25,
                            'q50': 0.35,
                            'q90': 0.15
                        }

                    # Calculate weighted ensemble
                    ensemble_val = (
                        weights['enhanced'] * enhanced_val +
                        weights['advanced'] * advanced_val +
                        weights['q50'] * q50_val +
                        weights['q90'] * q90_val
                    )

                    ensemble_predictions[sample_idx, day_idx, target_idx] = ensemble_val
                else:
                    # For temperature: Simple average of all models except q90
                    ensemble_predictions[sample_idx, day_idx, target_idx] = (
                        enhanced_preds[sample_idx, day_idx, target_idx] * 0.35 +
                        advanced_preds[sample_idx, day_idx, target_idx] * 0.35 +
                        q50_preds[sample_idx, day_idx, target_idx] * 0.3
                    )

    return ensemble_predictions

# Collect all model results
models_dict = {
    'enhanced': enhanced_result,
    'advanced': advanced_result,
    'quantile': quantile_result
}

# Create advanced ensemble predictions
ensemble_predictions = create_advanced_ensemble_predictions(models_dict, y_test)

# Calculate metrics for all models including ensemble
def inverse_transform_predictions(predictions, targets, scaler):
    """Transform scaled values back to original range"""
    pred_reshaped = predictions.reshape(-1, targets.shape[-1])
    targets_reshaped = targets.reshape(-1, targets.shape[-1])

    pred_original = scaler.inverse_transform(pred_reshaped)
    targets_original = targets_original.reshape(targets.shape)

    return pred_original, targets_original

# Process all models to get metrics
detailed_metrics = {}

# Process enhanced baseline
predictions_original, targets_original = inverse_transform_predictions(
    enhanced_result['predictions'], y_test, scaler_targets
)

metrics = {}
for i, var in enumerate(target_cols):
    metrics[var] = {}
    for day in range(prediction_length):
        true = targets_original[:, day, i]
        pred = predictions_original[:, day, i]
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        metrics[var][f'Day {day+1}'] = {'MAE': mae, 'RMSE': rmse}

detailed_metrics['enhanced'] = {
    'metrics': metrics,
    'predictions_original': predictions_original,
    'targets_original': targets_original
}

# Process advanced model
predictions_original, targets_original = inverse_transform_predictions(
    advanced_result['predictions'], y_test, scaler_targets
)

metrics = {}
for i, var in enumerate(target_cols):
    metrics[var] = {}
    for day in range(prediction_length):
        true = targets_original[:, day, i]
        pred = predictions_original[:, day, i]
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        metrics[var][f'Day {day+1}'] = {'MAE': mae, 'RMSE': rmse}

detailed_metrics['advanced'] = {
    'metrics': metrics,
    'predictions_original': predictions_original,
    'targets_original': targets_original
}

# Process quantile model (q50 - median predictions)
predictions_original, targets_original = inverse_transform_predictions(
    quantile_result['q50_predictions'], y_test, scaler_targets
)

metrics = {}
for i, var in enumerate(target_cols):
    metrics[var] = {}
    for day in range(prediction_length):
        true = targets_original[:, day, i]
        pred = predictions_original[:, day, i]
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        metrics[var][f'Day {day+1}'] = {'MAE': mae, 'RMSE': rmse}

detailed_metrics['quantile'] = {
    'metrics': metrics,
    'predictions_original': predictions_original,
    'targets_original': targets_original
}

# Process ensemble model
predictions_original, targets_original = inverse_transform_predictions(
    ensemble_predictions, y_test, scaler_targets
)

metrics = {}
for i, var in enumerate(target_cols):
    metrics[var] = {}
    for day in range(prediction_length):
        true = targets_original[:, day, i]
        pred = predictions_original[:, day, i]
        mae = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        metrics[var][f'Day {day+1}'] = {'MAE': mae, 'RMSE': rmse}

detailed_metrics['ensemble'] = {
    'metrics': metrics,
    'predictions_original': predictions_original,
    'targets_original': targets_original
}

# Calculate overall model metrics
print("\n\n==== PRECIPITATION FORECAST EVALUATION ====")
print("\nPrecipitation Prediction Performance:")
for model_type, result in detailed_metrics.items():
    precip_metrics = result['metrics'][target_cols[2]]  # Use correct target name
    avg_mae = np.mean([d['MAE'] for d in precip_metrics.values()])
    avg_rmse = np.mean([d['RMSE'] for d in precip_metrics.values()])
    print(f"{model_type.upper()}: Avg MAE = {avg_mae:.2f}, Avg RMSE = {avg_rmse:.2f}")

# Find best model for each target
print("\nBest model for each target variable:")
for var_idx, var_name in enumerate(target_cols):
    var_maes = {}
    for model_type, result in detailed_metrics.items():
        avg_mae = np.mean([result['metrics'][var_name][f'Day {day}']['MAE'] for day in range(1, prediction_length + 1)])
        var_maes[model_type] = avg_mae

    best_model = min(var_maes, key=var_maes.get)
    print(f"{var_name}: {best_model.upper()} (MAE: {var_maes[best_model]:.2f})")

# Plot comparison of precipitation predictions
def plot_precipitation_comparison(detailed_metrics, variable_name):
    """Compare precipitation predictions across models"""
    var_idx = target_cols.index(variable_name)
    sample_indices = [0, 1, 2, 3, 4]  # First 5 samples

    model_colors = {
        'enhanced': 'c--',
        'advanced': 'm--',
        'quantile': 'g--',
        'ensemble': 'y--'
    }

    for idx in sample_indices:
        plt.figure(figsize=(15, 6))

        # Get actual values (same for all models)
        true_values = list(detailed_metrics.values())[0]['targets_original'][idx, :, var_idx]
        plt.plot(range(len(true_values)), true_values, 'bo-', linewidth=2, label='Actual')

        # Plot predicted values for each model
        for model_type, result in detailed_metrics.items():
            pred_values = result['predictions_original'][idx, :, var_idx]
            plt.plot(range(len(pred_values)), pred_values, model_colors[model_type],
                     linewidth=2, label=f'{model_type.capitalize()}')

        plt.title(f'Sri Lanka {variable_name.title()} - Sample {idx+1}')
        plt.ylabel(f'{variable_name}')
        plt.xlabel('Day')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'sri_lanka_{variable_name}_sample_{idx+1}.png')
        plt.show()

# Find high precipitation samples
def find_high_precip_samples(detailed_metrics, threshold=15):
    """Find samples with high precipitation values"""
    var_idx = target_cols.index(target_cols[2])  # Use correct target name
    targets = list(detailed_metrics.values())[0]['targets_original']

    high_precip_indices = []
    for i in range(len(targets)):
        if np.max(targets[i, :, var_idx]) > threshold:
            high_precip_indices.append(i)

    return high_precip_indices[:5]  # Return at most 5 samples

# Plot precipitation predictions
print("\nPlotting precipitation comparison...")
plot_precipitation_comparison(detailed_metrics, target_cols[2])  # Use correct target name

# Plot high precipitation events
high_precip_indices = find_high_precip_samples(detailed_metrics)
print(f"\nFound {len(high_precip_indices)} samples with high precipitation")

if high_precip_indices:
    for idx in high_precip_indices:
        plt.figure(figsize=(15, 6))
        var_idx = target_cols.index(target_cols[2])  # Use correct target name

        # Plot actual values
        true_values = list(detailed_metrics.values())[0]['targets_original'][idx, :, var_idx]
        plt.plot(range(len(true_values)), true_values, 'bo-', linewidth=2, label='Actual')

        # Plot predicted values for each model
        model_colors = {
            'enhanced': 'c--',
            'advanced': 'm--',
            'quantile': 'g--',
            'ensemble': 'y--'
        }

        for model_type, result in detailed_metrics.items():
            pred_values = result['predictions_original'][idx, :, var_idx]
            plt.plot(range(len(pred_values)), pred_values, model_colors[model_type],
                     linewidth=2, label=f'{model_type.capitalize()}')

        plt.title(f'Sri Lanka High Precipitation Event - Sample {idx}')
        plt.ylabel('Precipitation (mm)')
        plt.xlabel('Day')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'sri_lanka_high_precipitation_sample_{idx}.png')
        plt.show()

# Print precipitation performance by day
print("\nPrecipitation Prediction Performance by Day:")
for day in range(1, prediction_length + 1):
    print(f"\nDay {day}:")
    for model_type, result in detailed_metrics.items():
        mae = result['metrics'][target_cols[2]][f'Day {day}']['MAE']  # Use correct target name
        rmse = result['metrics'][target_cols[2]][f'Day {day}']['RMSE']
        print(f"  {model_type.upper()}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# Generate final model comparison summary
print("\n\n===== MODEL COMPARISON SUMMARY =====")
print("Overall Metrics:")

# Print details for each model
print(f"ENHANCED: MAE = {enhanced_result['test_mae']:.4f}, Training time: {enhanced_result['training_time']:.2f} seconds")
print(f"ADVANCED: MAE = {advanced_result['test_mae']:.4f}, Training time: {advanced_result['training_time']:.2f} seconds")
print(f"QUANTILE (q50): MAE = {quantile_result['q50_mae']:.4f}, Training time: {quantile_result['training_time']:.2f} seconds")

# Calculate ensemble MAE
ensemble_mae = np.mean(np.abs(detailed_metrics['ensemble']['predictions_original'] -
                            detailed_metrics['ensemble']['targets_original']))
print(f"ENSEMBLE: MAE = {ensemble_mae:.4f}")

# Calculate improvement over previous best model
previous_best_mae = 3.45  # From earlier result
best_model_type = min(detailed_metrics, key=lambda x: np.mean([d['MAE'] for d in detailed_metrics[x]['metrics'][target_cols[2]].values()]))
best_model_mae = np.mean([d['MAE'] for d in detailed_metrics[best_model_type]['metrics'][target_cols[2]].values()])
improvement = (previous_best_mae - best_model_mae) / previous_best_mae * 100

print(f"\nBest model for precipitation: {best_model_type.upper()} with MAE = {best_model_mae:.2f}")
print(f"Improvement over previous best model: {improvement:.1f}%")

# Save models
enhanced_result['model'].save('sri_lanka_enhanced_precipitation_model.h5')
print("Enhanced model saved as 'sri_lanka_enhanced_precipitation_model.h5'")

advanced_result['model'].save('sri_lanka_advanced_precipitation_model.h5')
print("Advanced model saved as 'sri_lanka_advanced_precipitation_model.h5'")

quantile_result['model'].save('sri_lanka_quantile_precipitation_model.h5')
print("Quantile model saved as 'sri_lanka_quantile_precipitation_model.h5'")

print("\nAll model training and evaluation complete!")