import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Input, Conv1D, MaxPooling1D, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import gc
from datetime import datetime
import joblib

# Set memory growth for GPU if available
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    print("No GPU acceleration available")

class WeatherPredictor:
    def __init__(self, data_path, target_column='precipitation_sum', sequence_length=7, 
                 batch_size=32, model_path='models'):
        """
        Initialize the WeatherPredictor class
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file containing weather data
        target_column : str
            Column name to predict
        sequence_length : int
            Number of time steps to use for sequence prediction
        batch_size : int
            Batch size for training
        model_path : str
            Path to save trained models
        """
        self.data_path = data_path
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Initialize TF session with memory limits
        self._configure_memory()
    
    def _configure_memory(self):
        """Configure TensorFlow to use memory efficiently"""
        # Limit memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory config error: {e}")
        
        # Limit TF memory usage to stay under 12GB
        tf.config.set_soft_device_placement(True)
        tf.config.threading.set_intra_op_parallelism_threads(6)
        tf.config.threading.set_inter_op_parallelism_threads(6)
    
    def load_data(self, chunk_size=1000):
        """
        Load data in chunks to conserve memory
        
        Parameters:
        -----------
        chunk_size : int
            Number of rows to load at a time
        
        Returns:
        --------
        pandas.DataFrame
            Processed weather data
        """
        print(f"Loading data from {self.data_path}...")
        
        # Check file size before loading
        file_size = os.path.getsize(self.data_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        
        # Use chunking for large files
        if file_size > 100:
            chunks = []
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
                chunks.append(chunk)
                gc.collect()  # Force garbage collection
            df = pd.concat(chunks)
        else:
            df = pd.read_csv(self.data_path)
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Process date/time columns
        try:
            df['date'] = pd.to_datetime(df['time'])
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
        except:
            print("Warning: Date processing failed or 'time' column not found")
        
        # Parse time columns 
        try:
            df['sunrise'] = pd.to_datetime(df['sunrise'], format='%I:%M:%S %p').dt.hour * 60 + \
                          pd.to_datetime(df['sunrise'], format='%I:%M:%S %p').dt.minute
            df['sunset'] = pd.to_datetime(df['sunset'], format='%I:%M:%S %p').dt.hour * 60 + \
                         pd.to_datetime(df['sunset'], format='%I:%M:%S %p').dt.minute
        except:
            print("Warning: Sunrise/sunset parsing failed or columns not found")
            
        # Handle missing values
        df = df.interpolate(method='linear')
        
        # Drop any remaining rows with NaN
        df = df.dropna()
        
        # Memory cleanup
        gc.collect()
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the data for modeling
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        
        Returns:
        --------
        tuple
            Processed X and y data, and feature names
        """
        print("Preprocessing data...")
        
        # Copy to avoid modifying the original
        data = df.copy()
        
        # Drop non-numeric and unnecessary columns
        cols_to_drop = ['time', 'date', 'country', 'city']
        for col in cols_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        # Convert categorical variables
        cat_cols = data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            data[col] = pd.factorize(data[col])[0]
        
        # Feature selection - using correlations to select most important features
        numeric_data = data.select_dtypes(include=[np.number])
        target_corrs = numeric_data.corrwith(numeric_data[self.target_column]).abs().sort_values(ascending=False)
        print(f"Top correlations with {self.target_column}:")
        print(target_corrs.head(10))
        
        # Select features with correlation above threshold
        threshold = 0.1
        selected_features = target_corrs[target_corrs > threshold].index.tolist()
        
        if self.target_column in selected_features:
            selected_features.remove(self.target_column)
        
        print(f"Selected {len(selected_features)} features")
        
        # Normalize features
        self.scalers['features'] = MinMaxScaler()
        self.scalers['target'] = MinMaxScaler()
        
        # Scale features and target
        X = self.scalers['features'].fit_transform(data[selected_features])
        y = self.scalers['target'].fit_transform(data[[self.target_column]])
        
        # Save feature names for later
        self.feature_names = selected_features
        
        return X, y, selected_features
    
    def create_sequences(self, X, y):
        """
        Create sequences for time series prediction
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature array
        y : numpy.ndarray
            Target array
        
        Returns:
        --------
        tuple
            X_seq, y_seq arrays with sequences
        """
        print("Creating sequences...")
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """Build an optimized LSTM model"""
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_bidirectional_model(self, input_shape):
        """Build a Bidirectional LSTM model"""
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_enhanced_model(self, input_shape):
        """Build an enhanced CNN-LSTM model"""
        inputs = Input(shape=input_shape)
        
        # CNN for feature extraction
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        
        # LSTM for sequence learning
        lstm_out = LSTM(64, return_sequences=False)(pool1)
        
        # Merge layers
        dense1 = Dense(32, activation='relu')(lstm_out)
        dropout = Dropout(0.2)(dense1)
        output = Dense(1)(dropout)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_hybrid_model(self, input_shape):
        """Build a hybrid multi-input model"""
        # Main sequential input
        seq_input = Input(shape=input_shape)
        
        # CNN Branch
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(seq_input)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=32, kernel_size=3, activation='relu')(cnn)
        cnn_flat = tf.keras.layers.Flatten()(cnn)
        
        # LSTM Branch
        lstm = LSTM(64, return_sequences=True)(seq_input)
        lstm = Dropout(0.2)(lstm)
        lstm = LSTM(32)(lstm)
        
        # Combine branches
        combined = Concatenate()([cnn_flat, lstm])
        dense = Dense(32, activation='relu')(combined)
        dropout = Dropout(0.2)(dense)
        output = Dense(1)(dropout)
        
        model = Model(inputs=seq_input, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_models(self, X, y, validation_split=0.2, epochs=50):
        """
        Train multiple models and evaluate their performance
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input sequences
        y : numpy.ndarray
            Target values
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        
        Returns:
        --------
        dict
            Dictionary of trained models and their histories
        """
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Define models to train
        model_builders = {
            'lstm': self.build_lstm_model,
            'bidirectional': self.build_bidirectional_model,
            'enhanced': self.build_enhanced_model,
            'hybrid': self.build_hybrid_model
        }
        
        results = {}
        histories = {}
        
        # Train each model
        for name, builder in model_builders.items():
            print(f"\nTraining {name} model...")
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Build and train model
            model = builder(input_shape)
            
            # Create checkpoint callback
            checkpoint = ModelCheckpoint(
                f"{self.model_path}/{name}_model.h5",
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            # Train with early stopping
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
            
            # Load best model
            model = tf.keras.models.load_model(f"{self.model_path}/{name}_model.h5")
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"{name} model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Store results
            self.models[name] = model
            histories[name] = history.history
            results[name] = {'mse': mse, 'mae': mae, 'r2': r2}
            
            # Save model
            model.save(f"{self.model_path}/{name}_model_final.h5")
        
        # Create ensemble model (average predictions)
        print("\nCreating ensemble model...")
        self.create_ensemble(X_test, y_test)
        
        return results, histories
    
    def create_ensemble(self, X_test, y_test):
        """Create an ensemble of the trained models"""
        predictions = []
        
        for name, model in self.models.items():
            predictions.append(model.predict(X_test))
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Evaluate ensemble
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Ensemble model - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Store ensemble results
        self.models['ensemble'] = {'predictions': ensemble_pred}
        return mse, mae, r2
    
    def predict(self, input_data, model_name='ensemble'):
        """
        Make predictions using the specified model
        
        Parameters:
        -----------
        input_data : numpy.ndarray
            Input data for prediction
        model_name : str
            Name of the model to use
        
        Returns:
        --------
        numpy.ndarray
            Predicted values in original scale
        """
        if model_name == 'ensemble':
            # For ensemble, average predictions from all models
            predictions = []
            for name, model in self.models.items():
                if name != 'ensemble':
                    predictions.append(model.predict(input_data))
            prediction = np.mean(predictions, axis=0)
        else:
            # Use specified model
            prediction = self.models[model_name].predict(input_data)
        
        # Inverse transform to original scale
        prediction_original = self.scalers['target'].inverse_transform(prediction)
        
        return prediction_original
    
    def visualize_predictions(self, X_test, y_test, sample_indices=None):
        """
        Visualize model predictions against actual values
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test input data
        y_test : numpy.ndarray
            Test target data
        sample_indices : list
            Indices of samples to visualize, if None will select random samples
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with visualizations
        """
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if name != 'ensemble':
                predictions[name] = model.predict(X_test)
        
        # Ensemble prediction
        predictions['ensemble'] = np.mean([pred for pred in predictions.values()], axis=0)
        
        # If no sample indices provided, select random ones
        if sample_indices is None:
            num_samples = min(5, len(X_test))
            sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = [axes]
        
        # Plot each sample
        for i, idx in enumerate(sample_indices):
            ax = axes[i]
            
            # Actual values
            actual = self.scalers['target'].inverse_transform(y_test[idx].reshape(-1, 1))[0][0]
            
            # Plot model predictions
            model_names = list(predictions.keys())
            x_pos = 0.5
            bar_width = 0.8 / (len(model_names) + 1)
            
            # Plot actual value
            ax.bar(x_pos, actual, bar_width, label='Actual', color='blue')
            
            # Plot each model's prediction
            for j, name in enumerate(model_names):
                pred = self.scalers['target'].inverse_transform(
                    predictions[name][idx].reshape(-1, 1)
                )[0][0]
                ax.bar(x_pos + (j+1)*bar_width, pred, bar_width, label=name)
            
            ax.set_title(f'Sample {idx} Predictions')
            ax.set_ylabel(f'{self.target_column}')
            ax.set_xticks([x_pos + i*bar_width for i in range(len(model_names)+1)])
            ax.set_xticklabels(['Actual'] + model_names)
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def visualize_time_series(self, X_test, y_test, num_samples=5):
        """Plot time series predictions for multiple samples"""
        # Select sample indices
        sample_indices = np.linspace(0, len(X_test)-1, num_samples, dtype=int)
        
        # Prepare plot
        fig, axs = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axs = [axs]
        
        # Colors for each model
        colors = {
            'lstm': 'red',
            'bidirectional': 'green',
            'enhanced': 'cyan',
            'hybrid': 'magenta',
            'ensemble': 'yellow'
        }
        
        # Plot for each sample
        for i, idx in enumerate(sample_indices):
            ax = axs[i]
            
            # Create sequence of actual values (past observations + current target)
            sequence = X_test[idx, :, 0]  # Assuming first feature is the target or related
            current_target = y_test[idx][0]
            
            # Invert scaling for display
            seq_orig = self.scalers['features'].inverse_transform(
                np.column_stack([sequence] + [X_test[idx, :, j] for j in range(1, X_test.shape[2])])
            )[:, 0]
            
            target_orig = self.scalers['target'].inverse_transform([[current_target]])[0][0]
            
            # Combine past and present for display
            time_points = np.arange(len(seq_orig) + 1)
            values = np.append(seq_orig, target_orig)
            
            # Plot actual data
            ax.plot(time_points, values, 'b-o', label='Actual')
            
            # Plot model predictions
            for name, model in self.models.items():
                if name != 'ensemble':
                    pred = model.predict(X_test[idx:idx+1])[0][0]
                    pred_orig = self.scalers['target'].inverse_transform([[pred]])[0][0]
                    ax.plot(time_points[-1], pred_orig, 'o', color=colors[name], label=name)
            
            # Plot ensemble
            predictions = [self.models[m].predict(X_test[idx:idx+1])[0][0] 
                          for m in self.models if m != 'ensemble']
            ensemble_pred = np.mean(predictions)
            ensemble_orig = self.scalers['target'].inverse_transform([[ensemble_pred]])[0][0]
            ax.plot(time_points[-1], ensemble_orig, 'o', color=colors['ensemble'], label='ensemble')
            
            ax.set_title(f'Sample {idx}')
            ax.set_xlabel('Time')
            ax.set_ylabel(self.target_column)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save_models(self):
        """Save all models and scalers to disk"""
        for name, model in self.models.items():
            if name != 'ensemble':  # Ensemble is not a Keras model
                model.save(f"{self.model_path}/{name}_final.h5")
        
        # Save scalers
        joblib.dump(self.scalers, f"{self.model_path}/scalers.pkl")
        
        # Save feature names
        with open(f"{self.model_path}/feature_names.txt", 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        print(f"Models and scalers saved to {self.model_path}/")
    
    def load_models(self):
        """Load all models and scalers from disk"""
        self.models = {}
        for model_name in ['lstm', 'bidirectional', 'enhanced', 'hybrid']:
            try:
                self.models[model_name] = tf.keras.models.load_model(
                    f"{self.model_path}/{model_name}_final.h5"
                )
            except:
                print(f"Could not load {model_name} model")
        
        # Load scalers
        try:
            self.scalers = joblib.load(f"{self.model_path}/scalers.pkl")
        except:
            print("Could not load scalers")
        
        # Load feature names
        try:
            with open(f"{self.model_path}/feature_names.txt", 'r') as f:
                self.feature_names = f.read().splitlines()
        except:
            print("Could not load feature names")
        
        print("Models and scalers loaded")


def main():
    # Example usage
    data_path = "weather_data.csv"
    predictor = WeatherPredictor(data_path, target_column='precipitation_sum', 
                                sequence_length=7, batch_size=32)
    
    # Load and preprocess data
    df = predictor.load_data()
    X, y, features = predictor.preprocess_data(df)
    
    # Create sequences
    X_seq, y_seq = predictor.create_sequences(X, y)
    
    # Train models
    results, histories = predictor.train_models(X_seq, y_seq, epochs=30)
    
    # Save models
    predictor.save_models()
    
    # Create visualizations
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    # Visualize predictions
    fig1 = predictor.visualize_predictions(X_test[:10], y_test[:10])
    fig1.savefig('prediction_comparison.png')
    
    # Visualize time series
    fig2 = predictor.visualize_time_series(X_test, y_test, num_samples=5)
    fig2.savefig('time_series_prediction.png')
    
    print("Analysis complete. Model evaluation results:")
    for model, metrics in results.items():
        print(f"{model}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()