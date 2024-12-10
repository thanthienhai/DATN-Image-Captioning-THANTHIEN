import tensorflow as tf
from keras import layers, models
from keras.applications import efficientnet

def get_cnn_model(image_size=(299, 299, 3)):
    base_model = efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=image_size)
    base_model.trainable = False
    x = layers.Reshape((-1, base_model.output.shape[-1]))(base_model.output)
    return models.Model(base_model.input, x)

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense = layers.Dense(dense_dim, activation="relu")
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, inputs, training=False):
        x = self.attention(inputs, inputs, inputs)  # Self-attention
        x = self.norm1(inputs + x)  # Residual connection + Layer Normalization
        return self.norm2(x + self.dense(x))  # Feed-forward + Residual connection

class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, sequence_length, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()
        
        self.out = layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs, encoder_outputs, training=False, mask=None):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        token_embeds = self.token_embeddings(inputs)
        position_embeds = self.position_embeddings(positions)
        x = token_embeds + position_embeds

        x1 = self.attention_1(query=x, value=x, key=x, attention_mask=mask)
        x = self.norm1(x + x1)

        x2 = self.attention_2(query=x, value=encoder_outputs, key=encoder_outputs)
        x = self.norm2(x + x2)

        x3 = self.ffn(x)
        x = self.norm3(x + x3)

        return self.out(x)

class ImageCaptioningModel(models.Model):
    def __init__(self, cnn_model, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model  # CNN dùng để trích xuất đặc trưng ảnh
        self.encoder = encoder      # Encoder để mã hóa đặc trưng ảnh
        self.decoder = decoder      # Decoder để sinh chú thích

    def call(self, images, captions, training=False):
        img_features = self.cnn_model(images)
        encoder_outputs = self.encoder(img_features, training=training)
        predictions = self.decoder(captions, encoder_outputs, training=training)
        return predictions

    def compile(self, optimizer, loss):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

    def train_step(self, data):
        images, captions = data

        with tf.GradientTape() as tape:
            predictions = self(images, captions[:, :-1], training=True)
            loss = self.loss_fn(captions[:, 1:], predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_loss_metric.update_state(loss)
        return {"loss": self.train_loss_metric.result()}

    def test_step(self, data):
        images, captions = data
        predictions = self(images, captions[:, :-1], training=False)
        loss = self.loss_fn(captions[:, 1:], predictions)

        self.val_loss_metric.update_state(loss)
        return {"val_loss": self.val_loss_metric.result()}

    @property
    def metrics(self):
        return [self.train_loss_metric, self.val_loss_metric]