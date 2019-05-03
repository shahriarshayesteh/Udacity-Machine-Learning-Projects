

from keras import layers, models, optimizers, regularizers
from keras import backend as K

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states1 = layers.Dense(units=60, kernel_regularizer=regularizers.l2(0.01))(states)
        net_states2 = layers.Activation("relu")(net_states1)

        net_states3 = layers.Dense(units=150, kernel_regularizer=regularizers.l2(0.01))(net_states2)
        net_states4 = layers.Activation("relu")(net_states3)

        net_states5 = layers.Dense(units=50, kernel_regularizer=regularizers.l2(0.01))(net_states4)
        net_states6 = layers.Activation("relu")(net_states5)

        # Add hidden layer(s) for action pathway
        net_actions1 = layers.Dense(units=60, kernel_regularizer=regularizers.l2(0.01))(actions)
        net_actions2 = layers.Activation("relu")(net_actions1)
        
        net_actions3 = layers.Dense(units=150, kernel_regularizer=regularizers.l2(0.01))(net_actions2)
        net_actions4 = layers.Activation("relu")(net_actions3)
        
        net_actions5 = layers.Dense(units=50, kernel_regularizer=regularizers.l2(0.01))(net_actions4)
        net_actions6 = layers.Activation("relu")(net_actions5)


        # Combine state and action pathways
        net1 = layers.Add()([net_states5, net_actions5])
        net2 = layers.Activation('relu')(net1)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net2)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)