# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RLOO (REINFORCE Leave-One-Out) learner."""

import os
from typing import Any, Dict, Optional
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
from grain import python as grain
import jax
from jax import sharding
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.rl.grpo import rloo_learner as rloo_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

Mesh = sharding.Mesh

_DUMMY_DATA = [
    'input string',
    'hello world',
    'My name',
    'hello there',
]


def reward_fn(completions, **kargs):
  return jnp.arange(len(completions))


def reward_fn_2(prompts, answer, **kargs):
  return jnp.arange(len(answer))


class MySource(grain.RandomAccessDataSource):

  def __init__(self, data=None, repeat=1):
    if data is None:
      data = _DUMMY_DATA
    self._data = data * repeat

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)


def _dummy_dataset(source=MySource(), batch_size: int = 1):
  return (
      grain.MapDataset.source(source)
      .batch(batch_size)
      .map(lambda x: {'prompts': x, 'answer': x})
  )


def setup(kwargs: Optional[Dict[str, Any]] = None):
  if kwargs is None:
    kwargs = {}
  vocab = tc.MockVocab()
  model = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
      rngs=nnx.Rngs(0),
  )
  original_variables = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
  ref_model = tc.ToyTransformer(
      config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
  )

  mesh = pxla.thread_resources.env.physical_mesh
  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine='vanilla',
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optax.sgd(1e-3),
          eval_every_n_steps=kwargs.get('eval_every_n_steps', 2),
          max_steps=10,
          gradient_accumulation_steps=kwargs.get(
              'gradient_accumulation_steps', None
          ),
      ),
      rollout_config=base_rollout.RolloutConfig(
          max_tokens_to_generate=10,
          max_prompt_length=256,
          kv_cache_size=1024,
      ),
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=model,
      reference=ref_model,
      tokenizer=vocab,
      cluster_config=cluster_config,
  )
  return rl_cluster, model, original_variables


class RLOOConfigTest(parameterized.TestCase):
  """Tests for RLOOConfig initialization and validation."""

  def test_rloo_config_defaults(self):
    config = rloo_lib.RLOOConfig(num_generations=4)
    self.assertEqual(config.algo_variant, 'rloo')
    self.assertEqual(config.advantage_estimator, 'rloo')
    self.assertEqual(config.loss_agg_mode, 'token-mean')
    self.assertEqual(config.policy_loss_fn, 'grpo')

  def test_rloo_config_custom_params(self):
    config = rloo_lib.RLOOConfig(
        num_generations=8,
        num_iterations=2,
        beta=0.1,
        epsilon=0.3,
        loss_agg_mode='sequence-mean-token-mean',
    )
    self.assertEqual(config.num_generations, 8)
    self.assertEqual(config.num_iterations, 2)
    self.assertEqual(config.beta, 0.1)
    self.assertEqual(config.epsilon, 0.3)
    self.assertEqual(config.loss_agg_mode, 'sequence-mean-token-mean')
    # These should always be fixed regardless of input.
    self.assertEqual(config.algo_variant, 'rloo')
    self.assertEqual(config.advantage_estimator, 'rloo')

  def test_rloo_config_rejects_single_generation(self):
    with self.assertRaises(ValueError):
      rloo_lib.RLOOConfig(num_generations=1)

  def test_rloo_config_no_kl_penalty(self):
    config = rloo_lib.RLOOConfig(num_generations=4, beta=0.0)
    self.assertEqual(config.beta, 0.0)


class RLOOAdvantageTest(parameterized.TestCase):
  """Tests for the RLOO advantage estimator function."""

  def test_basic_rloo_advantages(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Group 1: [1, 2, 3], Group 2: [4, 5, 6]
    # LOO baselines for group 1: [2.5, 2.0, 1.5]
    # LOO baselines for group 2: [5.5, 5.0, 4.5]
    advantages = grpo_lib.compute_rloo_advantages(rewards, num_generations=3)
    expected = np.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected)

  def test_rloo_advantages_single_generation_returns_zeros(self):
    rewards = np.array([1.0, 2.0])
    advantages = grpo_lib.compute_rloo_advantages(rewards, num_generations=1)
    np.testing.assert_allclose(advantages, np.zeros_like(rewards))

  def test_rloo_advantages_two_generations(self):
    # With 2 generations, LOO baseline for each is just the other sample.
    rewards = np.array([3.0, 7.0, 1.0, 5.0])
    # Group 1: [3, 7] -> LOO baselines: [7, 3] -> advantages: [-4, 4]
    # Group 2: [1, 5] -> LOO baselines: [5, 1] -> advantages: [-4, 4]
    advantages = grpo_lib.compute_rloo_advantages(rewards, num_generations=2)
    expected = np.array([-4.0, 4.0, -4.0, 4.0])
    np.testing.assert_allclose(advantages, expected)

  def test_rloo_advantages_equal_rewards(self):
    # When all rewards in a group are equal, advantages should be zero.
    rewards = np.array([5.0, 5.0, 5.0, 5.0])
    advantages = grpo_lib.compute_rloo_advantages(rewards, num_generations=2)
    np.testing.assert_allclose(advantages, np.zeros(4))

  def test_rloo_advantages_sum_to_zero_per_group(self):
    # RLOO advantages within each group should sum to zero.
    rng = np.random.default_rng(42)
    num_generations = 4
    num_groups = 3
    rewards = rng.standard_normal(num_groups * num_generations)
    advantages = grpo_lib.compute_rloo_advantages(
        rewards, num_generations=num_generations
    )
    reshaped = advantages.reshape(-1, num_generations)
    np.testing.assert_allclose(
        reshaped.sum(axis=-1), np.zeros(num_groups), atol=1e-6
    )

  def test_rloo_vs_grpo_different_advantages(self):
    # RLOO and GRPO should produce different advantage values.
    rewards = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    rloo_adv = grpo_lib.compute_rloo_advantages(rewards, num_generations=3)
    grpo_adv = grpo_lib.compute_advantages(rewards, num_generations=3)
    # They should not be identical (different baseline computation).
    self.assertFalse(np.allclose(rloo_adv, grpo_adv))

  def test_rloo_advantages_large_group(self):
    # With many samples, LOO baseline should be close to group mean.
    num_generations = 16
    rewards = np.arange(num_generations, dtype=np.float64)
    advantages = grpo_lib.compute_rloo_advantages(
        rewards, num_generations=num_generations
    )
    # Sum of advantages should be ~0.
    np.testing.assert_allclose(advantages.sum(), 0.0, atol=1e-10)


class RLOOLearnerTest(parameterized.TestCase):
  """Tests for the RLOO learner end-to-end training loop."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    num_cpus = int(os.environ.get('DEVICE_COUNTS', 2))
    chex.set_n_cpu_devices(num_cpus)
    cls.device_count = jax.device_count()

  @parameterized.named_parameters(
      dict(
          testcase_name='single_reward_fn',
          reward_fns=reward_fn,
      ),
      dict(
          testcase_name='multiple_reward_fns',
          reward_fns=[reward_fn, reward_fn_2],
      ),
  )
  def test_rloo_learner_trains(self, reward_fns):
    kwargs = {'eval_every_n_steps': 2}
    rl_cluster, model, original_variables = setup(kwargs)
    rl_cluster.with_external_metrics_logger(print)
    rloo_config = rloo_lib.RLOOConfig(
        num_generations=2,
        num_iterations=1,
    )
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=rloo_config,
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    self.assertFalse(learner.should_sync_weights)
    self.assertFalse(learner.can_enable_async_rollout)

    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    eval_ds = _dummy_dataset(batch_size=1)
    learner.train(train_ds, eval_ds)

    # Verify that model parameters were updated.
    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)
    self.assertEqual(learner._iter_steps, 10)  # max_steps

  def test_rloo_learner_no_kl(self):
    """RLOO with beta=0.0 should skip reference inference."""
    kwargs = {'eval_every_n_steps': 2}
    rl_cluster, model, original_variables = setup(kwargs)
    rloo_config = rloo_lib.RLOOConfig(
        num_generations=2,
        num_iterations=1,
        beta=0.0,
    )
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn,
        algo_config=rloo_config,
    )
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    learner.train(train_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_rloo_uses_correct_advantage_estimator(self):
    """Verify that the RLOO learner uses the 'rloo' advantage estimator."""
    rloo_config = rloo_lib.RLOOConfig(num_generations=4)
    self.assertEqual(rloo_config.advantage_estimator, 'rloo')
    self.assertEqual(rloo_config.algo_variant, 'rloo')

  def test_rloo_learner_with_multiple_iterations(self):
    kwargs = {'eval_every_n_steps': 5}
    rl_cluster, model, original_variables = setup(kwargs)
    rloo_config = rloo_lib.RLOOConfig(
        num_generations=2,
        num_iterations=2,
    )
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fn,
        algo_config=rloo_config,
    )
    train_ds = _dummy_dataset(MySource(repeat=10), batch_size=2)
    learner.train(train_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)


if __name__ == '__main__':
  absltest.main()
