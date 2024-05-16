import torch
import torch.nn as nn
from torch.optim import Adam
import vision_transformer as vits
from torch.nn.utils.rnn import pad_sequence
from finetune_dataloader import AudioDataset
from torch.cuda.amp import GradScaler, autocast


def collate_fn(batch):

    segment_length = 1024  # Length of each segment
    overlap = 256  # Overlap between segments

    processed_tensors = []
    labels = []

    for tensor, label in batch:
        start = 0
        while start + segment_length <= tensor.shape[0]:
            segment = tensor[start:start + segment_length]
            if segment.shape[0] < segment_length:
                padding_size = segment_length - segment.shape[0]
                segment = torch.nn.functional.pad(segment, (0, 0, 0, padding_size), mode='constant', value=0)
            processed_tensors.append(segment.unsqueeze(0))
            labels.append(label)
            start += (segment_length - overlap)  # Move start up by segment length minus overlap

    # Stack all the processed tensors and labels into batches
    batch_tensors = torch.stack(processed_tensors, dim=0).repeat(1, 3, 1, 1)
    batch_labels = torch.tensor(labels, dtype=torch.long)

    return batch_tensors, batch_labels

def modify_pretrained_model(model, num_classes=2):
    # Replace the classifier head
    model.head = nn.Linear(model.embed_dim, num_classes)
    return model

def load_pretrained_model(student, path):
    checkpoint = torch.load(path, map_location='cpu')
    student.load_state_dict(checkpoint['student'], strict=False)
    print("Pre-trained model loaded from", path)

print(f"GPU's available: {torch.cuda.device_count()}")

# Load your pretrained model (assuming it is loaded and assigned to `pretrained_model`)
model = vits.__dict__['vit_base'](num_classes=2)  # Adjust number of classes
load_pretrained_model(model, './ASiT_INet_AS2M_16Khz.pth')
updated_model = modify_pretrained_model(model, num_classes=2).to('cuda:0')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
updated_model = nn.DataParallel(updated_model)
updated_model.to(device)
print("Model loaded and moved to device.")



# Freezing parameters and unfreezing the head
for param in updated_model.module.parameters():
    param.requires_grad = False
updated_model.module.head.weight.requires_grad = True
updated_model.module.head.bias.requires_grad = True

optimizer = Adam(filter(lambda p: p.requires_grad, updated_model.module.parameters()), lr=2e-5)

# Number of samples per class
num_real_samples = 8
num_fake_samples = 56

# Calculate weights
# Weight for each class is inversely proportional to its frequency
total_samples = num_real_samples + num_fake_samples
weight_for_real = total_samples / (2.0 * num_real_samples)
weight_for_fake = total_samples / (2.0 * num_fake_samples)

# Create a tensor of weights for Cross Entropy Loss
class_weights = torch.tensor([weight_for_fake, weight_for_real], dtype=torch.float32)

# If you're using a GPU, you need to move the weights to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# Initialize the loss function with these weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

dataset = AudioDataset('./dataset.json', './AUDIO/')
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    # num_workers=1,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn
    )
print(f"Data loaded: there are {len(dataset)} images.")


# Training loop
def train(model, dataloader, epochs, accumulation_steps=4):
    print("Training now")
    scaler = GradScaler()  # Initialize the gradient scaler

    for epoch in range(epochs):
        model.zero_grad()  # Reset gradients; do this once at the start of an epoch
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()

            # Forward pass in autocast mode to perform operations in mixed precision
            with autocast():
                outputs = model(inputs, classify=True)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss to account for accumulation

            # Backward pass
            scaler.scale(loss).backward()  # Scale loss before backpropagation

            # Perform optimization step after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # Apply optimization step
                scaler.update()  # Update the scale for next iteration
                model.zero_grad()  # Reset gradients after update

            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item() * accumulation_steps:.4f}")


train(updated_model, dataloader, epochs=20, accumulation_steps=4)

# save fintuned model as "finetuned_model.pth"
torch.save(updated_model.state_dict(), "finetuned_model.pth")