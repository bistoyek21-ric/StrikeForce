import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class EnemyAgentNet(Model):
    def __init__(self, 
                 num_cell_states=100, 
                 embedding_dim=16, 
                 grid_size=11, 
                 num_actions=10, 
                 num_state_factors=4):
        """
        Args:
            num_cell_states: Total distinct cell states (0-99).
            embedding_dim: Size of the embedding vector for each cell.
            grid_size: The grid dimensions (11x11).
            num_actions: Number of discrete actions (10).
            num_state_factors: Number of scalar inputs (damage, HP, stamina, coordination).
        """
        super(EnemyAgentNet, self).__init__()

        # --- Spatial Branch ---
        # 1. Embed the grid cell IDs to a dense vector representation.
        # Keras by default uses "channels_last" ordering.
        self.embedding = layers.Embedding(input_dim=num_cell_states, output_dim=embedding_dim)
        
        # 2. Convolution layers to extract spatial features.
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.pool = layers.MaxPool2D(pool_size=2)  # Downsamples spatial dimensions
        
        # --- Global Factors Branch ---
        # Process the dynamic scalar inputs with two fully connected layers.
        self.fc_factors = Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu")
        ])
        
        # --- Fusion & Decision ---
        # After pooling, our grid branch produces an output of shape (batch, 5, 5, 64)
        # This flattens to 5 * 5 * 64 = 1600.
        self.flatten = layers.Flatten()
        self.fc_combined = Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(num_actions)  # Final output: raw scores (logits) for each action.
        ])
    
    def call(self, grid, factors):
        """
        Args:
            grid: Tensor of shape (batch, grid_size, grid_size) with integer cell IDs.
            factors: Tensor of shape (batch, num_state_factors) with dynamic scalar values.
        Returns:
            Tensor of shape (batch, num_actions) containing the action scores.
        """
        # --- Spatial Branch Processing ---
        # Embedding: (batch, 11, 11) --> (batch, 11, 11, embedding_dim)
        x = self.embedding(grid)
        # Convolutional layers (maintaining channels_last data format).
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # Now shape becomes (batch, ~5, ~5, 64); for grid_size = 11, typically (batch, 5, 5, 64)
        x = self.flatten(x)  # Flatten to shape (batch, 1600)
        
        # --- Global Factors Processing ---
        # Process scalar inputs through dense layers.
        f = self.fc_factors(factors)  # Shape: (batch, 32)
        
        # --- Fusion ---
        # Merge spatial features and scalar factors along the feature dimension.
        combined = tf.concat([x, f], axis=1)  # Resulting shape: (batch, 1600+32)
        output = self.fc_combined(combined)    # Final output: (batch, num_actions)
        return output

# --- Example Usage ---
if __name__ == "__main__":
    print("$#" * 21)
    print(tf.__version__)
    exit()
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs available:", gpus)
    exit()
    # Define parameters:
    batch_size = 8
    grid_size = 11
    num_actions = 10         # 10 possible actions
    num_state_factors = 4    # damage, HP, stamina, coordination

    # Instantiate the model.
    model = EnemyAgentNet(num_cell_states=100, embedding_dim=16, grid_size=grid_size,
                           num_actions=num_actions, num_state_factors=num_state_factors)
    
    # Create dummy inputs:
    # Grid: integers from 0 to 99; shape: (batch, 11, 11)
    dummy_grid = tf.random.uniform((batch_size, grid_size, grid_size), minval=0, maxval=100, dtype=tf.int32)
    # Factors: floating-point values for the 4 dynamic scalars; shape: (batch, 4)
    dummy_factors = tf.random.uniform((batch_size, num_state_factors))
    
    # Perform a forward pass.
    outputs = model(dummy_grid, dummy_factors)
    print("Output shape:", outputs.shape)  # Expected output: (8, 10)
