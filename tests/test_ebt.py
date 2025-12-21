import torch
import torch.nn.functional as F

from nanochat.gpt import GPTConfig, EBT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTOL = 1e-4
ATOL = 1e-4

def energy_optim(model, input, n_steps, opt_step_size):
    x = input.clone().requires_grad_(True)
    _, T, _ = x.shape
    S = T // 2

    with torch.set_grad_enabled(True):
        for _ in range(n_steps):
            energy = model(x)[:, S:, :].sum()
            grad = torch.autograd.grad(
                energy,
                x,
                create_graph=True,
            )[0]
            grad[:, :S, :] = 0
            x = x - opt_step_size * grad

    return x


def test_ebt_no_flash():
    config = GPTConfig(
        sequence_len=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        use_flash_attn=False
    )
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    target = torch.randn(B, T, D, device=DEVICE)
    model = EBT(config)
    model = model.to(DEVICE)

    pred = energy_optim(
        model,
        input,
        2,
        0.8
    )


def test_ebt_with_flash():
    config = GPTConfig(
        sequence_len=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        use_flash_attn=True
    )
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    target = torch.randn(B, T, D, device=DEVICE)
    model = EBT(config)
    model = model.to(DEVICE)

    pred = energy_optim(
        model,
        input,
        2,
        0.8
    )

    
def test_closeness():
    config = GPTConfig(
        sequence_len=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=256,
        use_flash_attn=False
    )
    B, T, D = 4, config.sequence_len, config.n_embd
    input = torch.randn(B, T, D, device=DEVICE)
    target = torch.randn(B, T//2, D, device=DEVICE)
    model = EBT(config)
    model = model.to(DEVICE)

    model.zero_grad()
    orig_pred = energy_optim(
        model,
        input,
        2,
        0.8
    )
    output_1 = orig_pred[:, T//2:, :]
    loss_1 = F.mse_loss(output_1, target)
    loss_1.backward()

    config.use_flash_attn = True
    flash_model = EBT(config)
    flash_model.load_state_dict(model.state_dict())
    flash_model = flash_model.to(DEVICE)

    flash_model.zero_grad()
    flash_pred = energy_optim(
        flash_model,
        input,
        2,
        0.8
    )
    output_2 = flash_pred[:, T//2:, :]
    loss_2 = F.mse_loss(output_2, target)
    loss_2.backward()

    assert torch.allclose(orig_pred, flash_pred, rtol=RTOL, atol=ATOL), "Final optimization outputs are different"

    
    param1 = next(model.parameters())
    param2 = next(flash_model.parameters())
    assert torch.allclose(param1.grad, param2.grad, rtol=RTOL, atol=ATOL), "Gradients are different"
