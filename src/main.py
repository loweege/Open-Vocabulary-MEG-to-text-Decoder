import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from models.brain_translator_trainer import BrainTranslator_trainer
from dataset.load_dataset import load_dataset

def main():    
    train = False
    config = {
        "patches_number": 20,
        "channels": 208,
        "out_channels": 64,
        "in_feature": 768,
        "additional_encoder_nhead": 8, 
        "endoder_num_layers": 6,
        "additional_encoder_dim_feedforward": 2048,

        "batch_size": 8,
        "lr": 1e-5,
        "n_workers": 4
    }  

    train_dataloader, val_dataloader, test_dataloader = load_dataset(patches_number=config["patches_number"],
                    batch_size=config["batch_size"],
                    n_workers=config["n_workers"])
    
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints', save_top_k=2, monitor="val_loss")
    trainer = L.Trainer(accelerator="gpu",
                        devices=1, 
                        #strategy="ddp",
                        check_val_every_n_epoch=1,
                        max_epochs=50,
                        precision=16,
                        default_root_dir='./checkpoints',
                        callbacks=[checkpoint_callback])

    if train:
        model = BrainTranslator_trainer(config)
        trainer.fit(model, 
                    train_dataloader, 
                    val_dataloader)
    else:
        model = BrainTranslator_trainer.load_from_checkpoint("./checkpoints/epoch=26-step=53460.ckpt",
                                                             config)
        
    trainer.test(model, test_dataloader)

    import numpy as np
    results = {
        'rouge-1-r': np.array([x['rouge-1-r'] for x in model.predictions]).mean(),
        'rouge-1-p': np.array([x['rouge-1-p'] for x in model.predictions]).mean(),
        'rouge-1-f': np.array([x['rouge-1-f'] for x in model.predictions]).mean(),
        'bleu-1': np.array([x['bleu-1'] for x in model.predictions]).mean(),
        'bleu-2': np.array([x['bleu-2'] for x in model.predictions]).mean(),
        'bleu-3': np.array([x['bleu-3'] for x in model.predictions]).mean(),
        'bleu-4': np.array([x['bleu-4'] for x in model.predictions]).mean(),
    }
    print(results)

if __name__ == "__main__":
    main()




