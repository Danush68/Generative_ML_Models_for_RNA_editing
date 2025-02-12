class BitDiffusion(nn.Module):
    def __init__(self, seq_length=50, num_timesteps=1000):
        super().__init__()
        self.timesteps = num_timesteps
        self.noise_schedule = torch.linspace(1e-4, 0.02, num_timesteps)

        # U-Net for denoising
        self.denoiser = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3),  # Input: one-hot [batch, 4, seq]
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 4, kernel_size=3)
        )

    def forward(self, x_noisy, t, target_cond):
        # x_noisy: Noisy gRNA sequence [batch, 4, seq]
        # target_cond: Target sequence embedding
        return self.denoiser(x_noisy)

    def generate(self, target_seq, desired_efficiency):
        # Sample noise and denoise over timesteps
        x = torch.randn(4, len(target_seq))
        for t in reversed(range(self.timesteps)):
            x = self.denoise_step(x, t, target_seq)
        return x