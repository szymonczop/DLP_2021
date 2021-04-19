import argparse
import models_experiments as models
import json
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from data_augmentation import train_generator, test_generator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--model", default="mlp_model1", help="Model architecture", type=str, required=True,
                        choices=["mlp_model1", "mlp_model2", "mlp_model3", "mlp_model4", "mlp_model5",
                                 "cnn_model1", "cnn_model2", "cnn_model3", "cnn_model4", "cnn_model5"])
    parser.add_argument("--params", help="Path to JSON config file", default=None, required=False)
    parser.add_argument("--logs", default='logs/model', help="Path to logs", type=str)
    parser.add_argument("--checkpoint", default='model.h5', help="Name of checkpoint file")
    parser.add_argument("--lr", default=0.01, help="Learning rate", type=float)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--early_stop_patience", default=5, type=int)
    parser.add_argument("--reduce_lr_patience", default=3, type=int)

    args = parser.parse_args()
    return args


args = parse_arguments()

model_dict = {"mlp_model1": models.mlp_model1(),
              "mlp_model2": models.mlp_model2(),
              "mlp_model3": models.mlp_model3(),
              "mlp_model4": models.mlp_model4(),
              "mlp_model5": models.mlp_model5(),
              "cnn_model1": models.cnn_model1(),
              "cnn_model2": models.cnn_model2(),
              "cnn_model3": models.cnn_model3()
              # "cnn_model4": models.cnn_model4(),
              # "cnn_model5": models.cnn_model5()
              }


selected_model = model_dict[args.model]
selected_model.summary()

train_samples = 28709
test_samples = 7178
batch_size = 64

if args.params is None:
    early_stopping = EarlyStopping(monitor="val_loss", verbose=1, patience=args.early_stop_patience)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, patience=args.reduce_lr_patience)
    tensorboard = TensorBoard(args.logs)
    model_checkpoint = ModelCheckpoint(args.checkpoint, save_best_only=True)

    history = selected_model.fit(
        train_generator, steps_per_epoch = train_samples//batch_size, epochs=args.num_epochs,
        validation_data=test_generator, validation_steps = test_samples//batch_size, shuffle=True,
        callbacks=[early_stopping, tensorboard, reduce_lr, model_checkpoint]
    )
else:
    # read model parameters
    with open(args.params) as f:
        model_params = json.load(f)

    early_stopping = EarlyStopping(monitor="val_loss", verbose=1, patience=model_params["early_stop_patience"])
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", verbose=1, patience=model_params["reduce_lr_patience"])
    tensorboard = TensorBoard(model_params['logs'])
    model_checkpoint = ModelCheckpoint(model_params['checkpoint'], save_best_only=True)

    history = selected_model.fit(
        train_generator, steps_per_epoch = train_samples//batch_size, epochs=model_params["num_epochs"],
        validation_data=test_generator, validation_steps = test_samples//batch_size, shuffle=True,
        callbacks=[early_stopping, tensorboard, reduce_lr, model_checkpoint]
    )
