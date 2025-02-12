def activation_maximization(model, target_seq, num_steps=1000, lr=0.1):
    # Initialize a soft gRNA sequence (continuous relaxation)
    soft_gRNA = torch.randn(len(target_seq), requires_grad=True)
    optimizer = torch.optim.Adam([soft_gRNA], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        # Convert softmax to one-hot
        one_hot = torch.softmax(soft_gRNA, dim=-1)
        pred = model(one_hot, target_seq)
        loss = -pred[:, 0]  # Maximize efficiency
        loss.backward()
        optimizer.step()

    # Discretize final sequence
    hard_gRNA = torch.argmax(soft_gRNA, dim=-1)
    return hard_gRNA