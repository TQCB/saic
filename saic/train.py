from torch import nn, optim

from compression import HyperpriorCheckerboardCompressor
from loss import SAICLoss

def main():
    model = HyperpriorCheckerboardCompressor(
        n=128,
        m=256,
        z_alphabet_size=201,
    )
    criterion = SAICLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    epochs = 10
    outputs = []
    losses = []
    for epoch in range(epochs):
        for image, mask in loader:
            output = model(image, mask)
            loss = criterion(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        outputs.append((epoch, image, output['x_hat']))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")