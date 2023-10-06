import functools
import math
import jax.numpy as jnp

from lasi import LASI

partial = functools.partial

def test_compute_l1_distance():
  shape = (2, 1, 3)
  neighborhood_size = 3
  lasi = LASI(shape=shape, neighborhood_size=neighborhood_size)
  idx_points_coord = lasi.flat_idx_to_coordinates(
      jnp.arange(jnp.prod(jnp.array(shape)))
  )

  assert idx_points_coord.shape == (6, 3)
  assert (idx_points_coord[0] == jnp.array([0, 0, 0])).all()
  assert (idx_points_coord[1] == jnp.array([0, 0, 1])).all()
  assert (idx_points_coord[2] == jnp.array([0, 0, 2])).all()
  assert (idx_points_coord[3] == jnp.array([1, 0, 0])).all()
  assert (idx_points_coord[4] == jnp.array([1, 0, 1])).all()
  assert (idx_points_coord[5] == jnp.array([1, 0, 2])).all()

  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(0), idx_points_coord
      )
      == jnp.array([0, 1, 2, 1, 2, 3])
  ).all()
  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(1), idx_points_coord
      )
      == jnp.array([1, 0, 1, 2, 1, 2])
  ).all()
  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(2), idx_points_coord
      )
      == jnp.array([2, 1, 0, 3, 2, 1])
  ).all()
  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(3), idx_points_coord
      )
      == jnp.array([1, 2, 3, 0, 1, 2])
  ).all()
  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(4), idx_points_coord
      )
      == jnp.array([2, 1, 2, 1, 0, 1])
  ).all()
  assert (
      lasi.compute_l1_distance(
          lasi.flat_idx_to_coordinates(5), idx_points_coord
      )
      == jnp.array([3, 2, 1, 2, 1, 0])
  ).all()

def test_compute_l1_mask_idxs():
  shape = (2, 1, 3)
  lasi = LASI(shape=shape, neighborhood_size=2)
  mask_idx = lasi.compute_all_l1_mask_idxs()
  assert mask_idx.shape == (6, 2)
  assert (mask_idx[0] == jnp.array([-1, -1])).all()
  assert (mask_idx[1] == jnp.array([0, -1])).all()
  assert (mask_idx[2] == jnp.array([1, 0])).all()
  assert (mask_idx[3] == jnp.array([0, 1])).all()
  assert (mask_idx[4] == jnp.array([1, 3])).all()
  assert (mask_idx[5] == jnp.array([2, 4])).all()

def test_params_and_predict_element():
  shape = (5, 5)
  neighborhood_size = 3
  element_idx_flat = 0
  lasi = LASI(shape=shape, neighborhood_size=neighborhood_size)

  num_elements = math.prod(shape)
  tensor = (
      2 * jnp.arange(num_elements).reshape(shape) / (num_elements - 1) - 1
  )

  mask_idx = lasi.all_mask_idxs[element_idx_flat]
  tensor_transformed = lasi.transform_tensor(tensor)
  params = lasi.compute_element_params(element_idx_flat, tensor_transformed)

  assert lasi.predict_element(mask_idx, tensor, params) == 0

def test_predict_tensor():
  shape = (5, 5)
  neighborhood_size = 3
  lasi = LASI(shape=shape, neighborhood_size=neighborhood_size)
  num_elements = math.prod(shape)
  tensor = (
      2 * jnp.arange(num_elements).reshape(shape) / (num_elements - 1) - 1
  )
  tensor_transformed = lasi.transform_tensor(tensor)
  tensor_params = lasi.compute_tensor_params(tensor_transformed)
  tensor_pred = lasi.predict_tensor(tensor, tensor_params)

  assert tensor_pred.shape == tensor.shape
