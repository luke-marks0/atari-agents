from argparse import ArgumentParser
from functools import partial
from gzip import GzipFile
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import h5py
from ale_env import ALEModern, ALEClassic


class AtariNet(nn.Module):
    """ Estimator used by DQN-style algorithms for ATARI games. """
    def __init__(self, action_no, distributional=False):
        super().__init__()
        self.action_no = out_size = action_no
        self.distributional = distributional

        if distributional:
            support = torch.linspace(-10, 10, 51)
            self.__support = nn.Parameter(support, requires_grad=False)
            out_size = action_no * len(self.__support)

        self.__features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.__head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True), nn.Linear(512, out_size),
        )

    def forward(self, x):
        assert x.dtype == torch.uint8, "The model expects states of type ByteTensor"
        x = x.float().div(255)
        x = self.__features(x)
        qs = self.__head(x.view(x.size(0), -1))

        if self.distributional:
            logits = qs.view(qs.shape[0], self.action_no, len(self.__support))
            qs_probs = torch.softmax(logits, dim=2)
            return torch.mul(qs_probs, self.__support.expand_as(qs_probs)).sum(2)
        return qs


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)


def _epsilon_greedy(obs, model, eps=0.001):
    if torch.rand((1,)).item() < eps:
        return torch.randint(model.action_no, (1,)).item(), None
    q_val, argmax_a = model(obs).max(1)
    return argmax_a.item(), q_val


def collect_demonstrations(env, model, num_episodes, clip_length=25):
    """Collect demonstration data from expert policy"""
    states = []
    rewards = []
    actions = []

    with tqdm(total=num_episodes, desc="Collecting demonstrations") as pbar:
        for ep in range(num_episodes):
            episode_states = []
            episode_rewards = []
            episode_actions = []

            obs, done = env.reset(), False
            while not done:
                episode_states.append(obs.cpu().numpy())
                action, _ = _epsilon_greedy(obs, model, eps=0.001)
                episode_actions.append(action)
                obs, reward, done, _ = env.step(action)
                episode_rewards.append(reward)

            for i in range(0, len(episode_states) - clip_length + 1, clip_length):
                states.append(np.stack(episode_states[i:i + clip_length]))
                rewards.append(np.array(episode_rewards[i:i + clip_length]))
                actions.append(np.array(episode_actions[i:i + clip_length]))

            pbar.update(1)
            pbar.set_postfix({'Reward': sum(episode_rewards)})

    return {
        'states': np.stack(states),
        'rewards': np.stack(rewards),
        'actions': np.stack(actions)
    }


def save_demonstrations(data, output_path):
    """Save demonstration data in HDF5 format"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('states', data=data['states'], compression='gzip')
        f.create_dataset('rewards', data=data['rewards'], compression='gzip')
        f.create_dataset('actions', data=data['actions'], compression='gzip')


def process_model(model_path: Path, num_episodes: int):
    """Process a single model file"""
    parts = model_path.parts
    game = parts[-3]
    model_type = parts[-4]

    print(f"\nProcessing {game} ({model_type})...")

    ALE = ALEModern if "modern" in model_type else ALEClassic
    try:
        env = ALE(
            game,
            torch.randint(100_000, (1,)).item(),
            sdl=False,
            device="cpu",
            clip_rewards_val=False
        )
    except Exception as e:
        print(f"Error setting up environment for {game}: {e}")
        return False

    try:
        model = AtariNet(env.action_space.n, distributional="C51" in model_type)
        ckpt = _load_checkpoint(model_path)
        model.load_state_dict(ckpt["estimator_state"])
    except Exception as e:
        print(f"Error loading model for {game}: {e}")
        return False

    try:
        demonstrations = collect_demonstrations(env, model, num_episodes)
    except Exception as e:
        print(f"Error collecting demonstrations for {game}: {e}")
        return False

    try:
        output_path = Path('demonstrations') / model_type / f"{game}_expert_demos.h5"
        save_demonstrations(demonstrations, output_path)
        print(f"Successfully saved demonstrations for {game} to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving demonstrations for {game}: {e}")
        return False


def main(opt):
    base_path = Path(opt.models_dir)
    model_pattern = "**/model_*.gz"
    model_files = list(base_path.glob(model_pattern))

    if not model_files:
        print(f"No model files found in {base_path}")
        return

    print(f"Found {len(model_files)} model files")

    results = []
    with tqdm(total=len(model_files), desc="Processing models") as pbar:
        for model_path in model_files:
            success = process_model(model_path, opt.episodes)
            results.append((model_path, success))
            pbar.update(1)

    print("\nProcessing Summary:")
    successful = [p for p, s in results if s]
    failed = [p for p, s in results if not s]

    print(f"\nSuccessfully processed {len(successful)} models:")
    for path in successful:
        print(f"  ✓ {path.relative_to(base_path)}")

    if failed:
        print(f"\nFailed to process {len(failed)} models:")
        for path in failed:
            print(f"  ✗ {path.relative_to(base_path)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="directory containing the model files"
    )
    parser.add_argument(
        "-e",
        "--episodes",
        default=1,
        type=int,
        help="number of episodes to collect per model"
    )
    main(parser.parse_args())
