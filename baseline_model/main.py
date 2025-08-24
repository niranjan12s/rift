# Train
model = GPTLanguageModel().to(DEVICE)

# Print the number of parameters in the Transformer model
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Transformer params: {count_params(model):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Create a directory to save model checkpoints
checkpoint_dir = "transformer_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Simple training loop using the DataLoader
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0
    # Use STEPS_PER_EPOCH to iterate through the data loader
    STEPS_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE # Calculate STEPS_PER_EPOCH for the Transformer model
    for i in range(STEPS_PER_EPOCH):
        try:
            xb, yb = next(iter(train_loader))
        except StopIteration:
            # Reset the data loader iterator if we reach the end before STEPS_PER_EPOCH
            train_loader_iter = iter(train_loader)
            xb, yb = next(train_loader_iter)

        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

    # Save model checkpoint after each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Note: Add a validation step here if a validation dataset is available.