import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
# from segnet import SegNet
from dataset import CarlaDataset
from torch.utils.data import DataLoader

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # UNet paper says this
NUM_EPOCHS = 25
NUM_WORKERS = 2
PIN_MEMORY = True
IMAGE_HEIGHT = 240 # For resizing
IMAGE_WIDTH = 320 # For resizing
TRAIN_IMAGE_DIR = "Data/Train/CameraRGB/"
TRAIN_MASK_DIR = "Data/Train/CameraSeg/"
VAL_IMG_DIR = "Data/Test/CameraRGB/"
VAL_MASK_DIR = "Data/Test/CameraSeg/"

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarlaDataset(
        imageDir=train_dir,
        maskDir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarlaDataset(
        imageDir=val_dir,
        maskDir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def train(loader, model, optimizer, loss_fn, scaler):

    loop  = tqdm(loader)

    epoch_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loop):

        data, targets = next(iter(loader))
        
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) 

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    return epoch_loss/len(loader)

def check_accuracy(loader, model, device="cpu"):

    model.eval()

    num_corrects=0
    num_pixels=0
    dice_score=0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_corrects += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Validation accuracy : {num_corrects/num_pixels *100:.2f}")
    print(f"Dice score : {dice_score/len(loader)}")
    model.train()

def main():

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        print(f'EPOCH : {epoch+1}/{NUM_EPOCHS}')
        
        mean_epoch_loss = train(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        model_PATH = f'trained_unet.pt'
        torch.save(checkpoint, model_PATH)

        # model = UNET()
        # model_state_dict = torch.load(model_PATH)
        # model.load_state_dict(model_state_dict)
        # model.train()

        print(f'EPOCH {epoch+1} Mean Loss : {mean_epoch_loss}')
        check_accuracy(val_loader, model, device=DEVICE)


if __name__=="__main__":
    main()