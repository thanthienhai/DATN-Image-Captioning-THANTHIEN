import os
import tensorflow as tf
from models.models import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from models.utils import preprocess_image, vectorization
from keras.optimizers.schedules import LearningRateSchedule
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

TRAIN_IMAGES_PATH = "path/to/train_images"
TRAIN_CAPTIONS_PATH = "path/to/train_captions.csv"
MODEL_SAVE_PATH = "./saved_model/image_captioning_model"

EMBED_DIM = 512
DENSE_DIM = 512
NUM_HEADS = 4
VOCAB_SIZE = 13000
SEQ_LENGTH = 24
BATCH_SIZE = 32
EPOCHS = 20

def load_data(images_path, captions_path):
    pass

def train_model(train_dataset, val_dataset):
    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=DENSE_DIM, num_heads=NUM_HEADS)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=DENSE_DIM, num_heads=NUM_HEADS, 
                                      sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    
    loss_fn = SparseCategoricalCrossentropy(from_logits=False, reduction="none")
    lr_schedule = LearningRateSchedule(1e-4)
    optimizer = Adam(learning_rate=lr_schedule)

    caption_model.compile(optimizer=optimizer, loss=loss_fn)

    caption_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True)
        ]
    )
    print("Model training complete.")
    return caption_model

def load_trained_model():
    return tf.keras.models.load_model(MODEL_SAVE_PATH)

def generate_caption(image_path, model):
    image = preprocess_image(image_path)

    image_features = model.cnn_model(image)

    encoder_output = model.encoder(image_features, training=False)

    caption = "<start>"
    for _ in range(SEQ_LENGTH - 1):
        tokenized_caption = vectorization([caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoder_output, training=False, mask=mask)
        predicted_id = tf.argmax(predictions[0, -1, :]).numpy()
        word = vectorization.get_vocabulary()[predicted_id]
        if word == "<end>":
            break
        caption += " " + word

    return caption.replace("<start>", "").strip()

if __name__ == "__main__":
    mode = input("Enter 'train' to train the model or 'inference' to generate captions: ").strip().lower()

    if mode == "train":
        train_dataset, val_dataset = load_data(TRAIN_IMAGES_PATH, TRAIN_CAPTIONS_PATH)
        
        model = train_model(train_dataset, val_dataset)
        print(f"Model saved at {MODEL_SAVE_PATH}")

    elif mode == "inference":
        model = load_trained_model()

        image_path = input("Enter the path to the image: ").strip()
        caption = generate_caption(image_path, model)
        print(f"Generated Caption: {caption}")

    else:
        print("Invalid option. Please enter 'train' or 'inference'.")