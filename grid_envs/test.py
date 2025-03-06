import torch
import numpy as np

class DummyEnv:
    grid_size = (3, 3)
    n_obs_types = 4
    # The following attributes are defined for completeness; they are not used in one_hot_to_obs.
    obs_agent_self = 10
    obs_agent_other = 11
    obs_agent_both = 12
    obs_plant = 13
    obs_stag = 14

class DummyWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def one_hot_to_obs_original(self, one_hot_obs):
        """
        Original implementation: converts a one-hot observation to a discrete grid.
        Assumes input is unflattened (shape: [H, W, n_obs_types]).
        """
        GRID_SIZE = self.env.grid_size
        one_hot_obs = torch.tensor(one_hot_obs, dtype=torch.float32, device=self.device)
        one_hot_obs = one_hot_obs.reshape((GRID_SIZE[0], GRID_SIZE[1], self.env.n_obs_types))
        obs = torch.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=torch.int32, device=self.device)
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                obs[x][y] = torch.argmax(one_hot_obs[x][y]).item()
        return obs

    def one_hot_to_obs_vectorized(self, one_hot_obs):
        """
        Vectorized inverse of obs_to_one_hot, which can handle both flattened and
        grid-shaped inputs, as well as batched observations.
        """
        GRID_SIZE = self.env.grid_size
        n_obs_types = self.env.n_obs_types

        # Ensure the input is a tensor on the proper device.
        if not torch.is_tensor(one_hot_obs):
            one_hot_obs = torch.tensor(one_hot_obs, dtype=torch.float32, device=self.device)

        # If the observation was flattened (1D for single or 2D for batch), reshape it.
        if one_hot_obs.dim() == 1:
            one_hot_obs = one_hot_obs.view(GRID_SIZE[0], GRID_SIZE[1], n_obs_types)
        elif one_hot_obs.dim() == 2 and one_hot_obs.shape[1] == GRID_SIZE[0] * GRID_SIZE[1] * n_obs_types:
            one_hot_obs = one_hot_obs.view(-1, GRID_SIZE[0], GRID_SIZE[1], n_obs_types)

        # Use vectorized argmax over the last dimension.
        return torch.argmax(one_hot_obs, dim=-1)

# ------------------ Test Cases ------------------

def test_one_hot_to_obs_single_unflattened():
    env = DummyEnv()
    device = "cpu"
    wrapper = DummyWrapper(env, device)
    
    # Create a one-hot observation of shape (3, 3, 4)
    one_hot_obs = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    ])
    expected = torch.tensor([
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0]
    ], dtype=torch.int32, device=device)

    out_orig = wrapper.one_hot_to_obs_original(one_hot_obs)
    out_vector = wrapper.one_hot_to_obs_vectorized(one_hot_obs)
    assert torch.equal(out_orig, expected), f"Original output {out_orig} != expected {expected}"
    assert torch.equal(out_vector, expected), f"Vectorized output {out_vector} != expected {expected}"
    print("test_one_hot_to_obs_single_unflattened passed.")

def test_one_hot_to_obs_single_flattened():
    env = DummyEnv()
    device = "cpu"
    wrapper = DummyWrapper(env, device)
    
    one_hot_obs = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    ])
    # Flatten the observation.
    one_hot_flat = one_hot_obs.reshape(-1)
    expected = torch.tensor([
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0]
    ], dtype=torch.int32, device=device)

    # The original function expects an unflattened observation.
    out_orig = wrapper.one_hot_to_obs_original(one_hot_obs)
    out_vector = wrapper.one_hot_to_obs_vectorized(one_hot_flat)
    assert torch.equal(out_orig, expected), f"Original output {out_orig} != expected {expected}"
    assert torch.equal(out_vector, expected), f"Vectorized output {out_vector} != expected {expected}"
    print("test_one_hot_to_obs_single_flattened passed.")

def test_one_hot_to_obs_batch_unflattened():
    env = DummyEnv()
    device = "cpu"
    wrapper = DummyWrapper(env, device)
    
    # Create two one-hot observations (each of shape (3, 3, 4))
    obs1 = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    ])
    obs2 = np.array([
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    ])
    batch = np.stack([obs1, obs2], axis=0)
    
    expected1 = torch.tensor([
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0]
    ], dtype=torch.int32, device=device)
    expected2 = torch.tensor([
        [1, 0, 2],
        [0, 3, 1],
        [1, 0, 3]
    ], dtype=torch.int32, device=device)
    expected_batch = torch.stack([expected1, expected2], dim=0)

    out_vector = wrapper.one_hot_to_obs_vectorized(batch)
    # Since the original function does not handle batches, apply it to each observation.
    out_orig1 = wrapper.one_hot_to_obs_original(obs1)
    out_orig2 = wrapper.one_hot_to_obs_original(obs2)
    out_orig_batch = torch.stack([out_orig1, out_orig2], dim=0)

    assert torch.equal(out_vector, expected_batch), f"Vectorized output {out_vector} != expected {expected_batch}"
    assert torch.equal(out_orig_batch, expected_batch), f"Original batch output {out_orig_batch} != expected {expected_batch}"
    print("test_one_hot_to_obs_batch_unflattened passed.")

def test_one_hot_to_obs_batch_flattened():
    env = DummyEnv()
    device = "cpu"
    wrapper = DummyWrapper(env, device)
    
    obs1 = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    ])
    obs2 = np.array([
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]],
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    ])
    batch = np.stack([obs1, obs2], axis=0)
    # Flatten each observation in the batch.
    batch_flat = batch.reshape(batch.shape[0], -1)
    
    expected1 = torch.tensor([
        [0, 1, 2],
        [3, 0, 1],
        [2, 3, 0]
    ], dtype=torch.int32, device=device)
    expected2 = torch.tensor([
        [1, 0, 2],
        [0, 3, 1],
        [1, 0, 3]
    ], dtype=torch.int32, device=device)
    expected_batch = torch.stack([expected1, expected2], dim=0)

    out_vector = wrapper.one_hot_to_obs_vectorized(batch_flat)
    # Use original function on unflattened observations.
    out_orig1 = wrapper.one_hot_to_obs_original(obs1)
    out_orig2 = wrapper.one_hot_to_obs_original(obs2)
    out_orig_batch = torch.stack([out_orig1, out_orig2], dim=0)

    assert torch.equal(out_vector, expected_batch), f"Vectorized batch flattened output {out_vector} != expected {expected_batch}"
    assert torch.equal(out_orig_batch, expected_batch), f"Original batch output {out_orig_batch} != expected {expected_batch}"
    print("test_one_hot_to_obs_batch_flattened passed.")

if __name__ == "__main__":
    test_one_hot_to_obs_single_unflattened()
    test_one_hot_to_obs_single_flattened()
    test_one_hot_to_obs_batch_unflattened()
    test_one_hot_to_obs_batch_flattened()