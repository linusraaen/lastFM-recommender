import torch

from src.model.towers import ItemTower, UserTower


def test_item_tower_output_is_l2_normalised():
    tower = ItemTower(feature_dim=12, hidden_dim=8, embed_dim=6)
    features = torch.randn(5, 12)
    out = tower(features)
    assert out.shape == (5, 6)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-5)


def test_user_tower_pools_context_and_normalises():
    embed_dim = 6
    tower = UserTower(embed_dim=embed_dim, hidden_dim=8)
    context = torch.randn(3, 4, embed_dim)  # B=3, H=4
    weights = torch.ones(3, 4)
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
    ], dtype=torch.float32)

    out = tower(context, weights, mask)
    assert out.shape == (3, embed_dim)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_user_tower_ignores_masked_out_context():
    embed_dim = 4
    tower = UserTower(embed_dim=embed_dim, hidden_dim=8)
    context = torch.randn(1, 3, embed_dim)
    weights = torch.ones(1, 3)

    full_mask = torch.tensor([[1.0, 1.0, 0.0]])
    # replacing the masked-out slot's values should not change the output
    altered_context = context.clone()
    altered_context[0, 2] = torch.randn(embed_dim)

    with torch.no_grad():
        out1 = tower(context, weights, full_mask)
        out2 = tower(altered_context, weights, full_mask)

    assert torch.allclose(out1, out2, atol=1e-6)
