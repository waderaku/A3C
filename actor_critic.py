import numpy as np
import tensorflow as tf


class ActorCritic(tf.keras.models.Model):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.common_layer = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_dim)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        self.actor_layer = tf.keras.layers.Dense(
            n_actions, activation='softmax')
        self.value_layer = tf.keras.layers.Dense(1, activation='linear')

        self.beta = 0.01

    def call(self, inputs):
        inputs = self.common_layer(inputs)
        policy = self.actor_layer(inputs)
        value = self.value_layer(inputs)
        return(policy, value)

    def calc_loss(self, state, action, advatnage, logits, v_pred, td_targets):
        actor_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        logits_entropy = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        actor_loss = actor_entropy(action, logits,
                                   sample_weight=tf.stop_gradient(advatnage))
        - self.beta * logits_entropy(logits, logits)
        mean_error = tf.keras.losses.MeanSquaredError()
        critic_loss = mean_error(
            v_pred, td_targets)

        return (actor_loss, critic_loss)

    def train(self, state, action, value):
        with tf.GradientTape() as tape:
            logits = self.model(state)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(action, value,
                                                                 from_logits=True,
                                                                 sample_weight=tf.stop_gradient(value))
            - self.beta * tf.keras.losses.CategoricalCrossentropy(
                logits, logits, from_logits=True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables))
        return loss
