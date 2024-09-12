import os
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model

class DrugTar:
    """A class for defining the DrugTar model."""
    def __init__(self, input_dim, hidden_dim ,dropout_rate=0.5, model_name="DrugTar"):
        """Initialize the DrugTar model with specified hyperparameters."""
        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        self.model = None  # Model will be defined when build_model() is called

    def build_model(self):
        """Builds the model."""
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Hidden layers
        hidden_layer = Dense(self.hidden_dim[0], kernel_regularizer='l2')(input_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        hidden_layer = Dropout(self.dropout_rate)(hidden_layer)
        
        hidden_layer = Dense(self.hidden_dim[1], kernel_regularizer='l2')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        hidden_layer = Dropout(self.dropout_rate)(hidden_layer)
        
        hidden_layer = Dense(self.hidden_dim[2], kernel_regularizer='l2')(hidden_layer)
        hidden_layer = BatchNormalization()(hidden_layer)
        hidden_layer = Activation('relu')(hidden_layer)
        
        # Output layer
        output_layer = Dense(1, activation='sigmoid')(hidden_layer)

        # Define the model
        self.model = Model(inputs=input_layer, outputs=output_layer, name=self.model_name)
        
    def save_model(self, file_path):
        """Saves the trained model to the specified file."""
        if self.model:
            self.model.save(file_path)
        else:
            print("Error: Model is not built yet.")

    def load_model(self, file_path):
        """Loads a saved model from the specified file."""
        if os.path.exists(file_path):
            self.model = load_model(file_path)
        else:
            print(f"Error: The file {file_path} does not exist.")

    def get_model(self):
        """Returns the model."""
        return self.model

