import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_weather_data(filepath):
    """Load historical weather data from file"""
    df = pd.read_csv(filepath)
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess weather data for the enhanced model"""
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Feature engineering specific for enhanced model
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # Normalize features
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col not in ['date', 'target']]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    return df, scaler

def create_enhanced_model(input_shape):
    """Create the enhanced weather forecasting model"""
    # Enhanced architecture with attention mechanism
    inputs = layers.Input(shape=input_shape)
    
    # First LSTM layer with more units
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.Dense(1)(x)
    attention = layers.Reshape((-1,))(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.Reshape((-1, 1))(attention)
    
    # Apply attention to the LSTM outputs
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
    # Dense layers for prediction with dropout
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_sequences(data, target_col, seq_length=24):
    """Prepare sequences for the model input"""
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data.iloc[i+seq_length][target_col])
    
    return np.array(X), np.array(y)

def train_enhanced_model(data_path, model_save_path, epochs=100):
    """Train only the enhanced model for better accuracy"""
    # Load and preprocess data
    df = load_weather_data(data_path)
    df, scaler = preprocess_data(df)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    target_col = 'target'
    
    # Create sequence data
    X, y = prepare_sequences(df[feature_cols], target_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create the enhanced model
    model = create_enhanced_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train the model with validation
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save training history visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model, scaler

if __name__ == "__main__":
    DATA_PATH = "weather_data.csv"  # Update with your data path
    MODEL_SAVE_PATH = "enhanced_weather_model.h5"
    
    model, scaler = train_enhanced_model(DATA_PATH, MODEL_SAVE_PATH)