"""LASI: A JAX implementation of the Linear Autoregressive Similarity Index.

Methods in `LASI` will usually come in pairs: a method that takes a tensor
element as input and another that takes the entire tensor itself. The latter
is a helper that vmaps the former over the tensor.

First, LASI maps the input tensor to a transformed domain
(see `LASI.transform_element`, `LASI.transform_tensor`).

Then, for each element of the tensor, the transformed tensor is used to compute
a coefficient matrix `A` and target vector `b` which define the least squares
problem discussed in the paper. Taking an inner product between the solution 
`w` and a subset of previous elements yields a prediction of the tensor element.
(see `LASI.compute_element_params`, `LASI.compute_tensor_params`)

The subset of previous elements used during prediction is referred to as the
"causal neighborhood". They correspond to the closest elements in coordinate space
with respect to the l1-distance of the element being predicted.
(See `LASI.compute_all_l1_mask_idxs` for a proper definition.)

LASI has only 1 hyperparameter, `neighborhood_size`, defining the size of
the causal neighborhood, which is equal to the dimensionality of the 
linear regression problem, as well as the final embedding dimensionality.

"""

import functools
import math
import jax
import jax.numpy as jnp

from typing import Sequence, Tuple, Optional

partial = functools.partial


class LASI:
    """LASI: A JAX implementation of the Linear Autoregressive Similarity Index.

    See the top of this file for more details on all parameters.

    Parameters:
      shape: the shape of the input tensors.
      neighborhood_size: controls the size of the causal neighborhood.
      ols_regularization_coef: controls the regularization of the OLS problem.
      unit_normalize_features: if true, normalizes the params (OLS solutions)
        when computing the jlixbawls distance metric.

    Attributes:
      all_elements_idx_flat: 1D-array of values in range 0 to `math.prod(shape)-1`
        representing the index of elements in `tensor.flatten()`
      all_elements_idx_coord: same as `all_elements_idx_flat` but indexed in
        `tensor`.
      all_mask_idxs: array with index values for the l1-neighborhood of all
        elements in `tensor`.
    """

    def __init__(
        self,
        shape: Sequence[int],
        neighborhood_size: int,
        ols_regularization_coef: float = 1.0,
        unit_normalize_features: bool = True,
    ):
        self.shape = shape
        self.neighborhood_size = neighborhood_size
        self.ols_regularization_coef = ols_regularization_coef
        self.unit_normalize_features = unit_normalize_features
        self.all_elements_idx_flat = jnp.arange(math.prod(shape))
        self.all_elements_idx_coord = self.flat_idx_to_coordinates(
            self.all_elements_idx_flat
        )
        self.all_mask_idxs = self.compute_all_l1_mask_idxs()

    def flat_idx_to_coordinates(self, idxs_flat: jnp.ndarray) -> jnp.ndarray:
        """Converts indices for flat tensors to those of shaped tensors.

        The following code illustrates the use.

        >>> tensor_flat = tensor.flatten()
        >>> idxs_flat = jnp.array([2, 4, 6])
        >>> idxs_coord = flat_idx_to_coordinates(tensor.shape, idxs_flat)
        >>> assert (tensor_flat[idxs_flat] == tensor[idxs_coord].flatten()).all()

        Args:
          idxs_flat: array of flat array indices to be converted

        Returns:
          An array with shape `(len(idxs_flat), len(shape))`.
        """
        return jnp.stack(jnp.unravel_index(idxs_flat, self.shape)).T

    def compute_l1_distance(self, idx_centre_coord, idxs_points_coord):
        return jnp.abs(idxs_points_coord - idx_centre_coord).sum(-1)

    def compute_all_l1_mask_idxs(self) -> jnp.ndarray:
        """Computes a `mask` defining the l1-neighborhoods of elements in a tensor.

        This function returns only `mask`, not the values of the neighborhood.
        To get the values one must flatten the tensor and then slice with `mask`.

        The l1-neighborhood is determined by the coordinates of the element, not the
        value. For example, if `shape=(3, 3)` and `neighborhood_size=3`, then the 
        causal neighborhood for coordinate `(1, 1)` will be 
                                
                                |       (0, 1) (0, 2)|
                                |(1, 0) (1, 1)       |
                                |                    |,
                
        except (1, 1) itself, which was added for the sake of illustration.

        When there aren't enough points to fill the neighborhood the value -1 is
        used. A default value is then imputed in LASI.get_neighborhood.

        The output indices in the mask are with respect to the flattened tensor. In
        the case of the previous example then `mask[4] = [3, 1, 2]`.

        Returns:
          A `mask` with shape `(math.prod(shape), neighborhood_size)` where
          `tensor_flat[mask[i]]` will return the values of the neighborhood of
          `tensor_flat[i]`.
        """
        total_num_elements = math.prod(self.shape)
        assert (
            total_num_elements > self.neighborhood_size
        ), "Total number of elements must be greater than the neighborhood size"

        flat_idx = jnp.arange(total_num_elements)
        idxs_coord = self.flat_idx_to_coordinates(flat_idx)

        @jax.jit
        def compute_l1_mask_single_element(idx_centre_flat):
            mask = jnp.ones_like(flat_idx).astype(bool)
            mask &= flat_idx < idx_centre_flat

            idx_centre_coord = self.flat_idx_to_coordinates(idx_centre_flat)
            l1_distance = self.compute_l1_distance(idx_centre_coord, idxs_coord)
            l1_distance_masked = jnp.where(mask, l1_distance, jnp.inf)
            sort_perm = jnp.argsort(l1_distance_masked)
            mask_idxs = jnp.where(
                l1_distance_masked[sort_perm] < jnp.inf, sort_perm, -1
            )
            return mask_idxs[: self.neighborhood_size]

        return jax.vmap(compute_l1_mask_single_element)(flat_idx)

    def get_element_neighborhood(
        self, mask_idx: jnp.array, tensor_flat: jnp.array, fill_value=0
    ) -> jnp.array:
        """Slices the tensor with the mask to return the causal neighborhood of elements.

        Args:
          mask_idx: a mask corresponding to the neighborhood of a single element.
            Usually `self.all_mask_idxs[flat_idx]` where `flat_idx` is the index of
            element (in the flat tensor) to which the neighborhood corresponds to.
          tensor_flat: flattened jax array of your tensor.
          fill_value: value to impute into missing neighborhood values.

        Returns:
          An array of shape `mask_idx.shape` containing the values of neighborhoods.
        """
        return jnp.where(mask_idx != -1, tensor_flat[mask_idx], fill_value)

    def transform_element(
        self, mask_idx: jnp.array, element_value: float, tensor_flat: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Applies the LASI transform to a single element of a tensor.

        Args:
          mask_idx: a mask corresponding to the neighborhood of a single element.
            Usually `self.all_mask_idxs[flat_idx]` where `flat_idx` is the index of
            element (in the flat tensor) to which the neighborhood corresponds to.
          element_value: value of the element to be transformed.
          tensor_flat: flattened jax array of your tensor.

        Returns:
          A tuple of 2 elements. One is used to compute the coefficient matrix,
          and the other the target, during prediction. Check the description at the
          top of this file for more details.
        """

        neighborhood = self.get_element_neighborhood(mask_idx, tensor_flat)
        return jnp.outer(neighborhood, neighborhood.T), element_value * neighborhood

    def transform_tensor(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Applies the LASI transform to a tensor.

        Args:
          tensor: your tensor.

        Returns:
          A tuple of 2 elements. One is used to compute the coefficient matrix,
          and the other the target, during prediction. Check the description at the
          top of this file for more details.
        """

        tensor_flat = tensor.flatten()
        return jax.vmap(partial(self.transform_element, tensor_flat=tensor_flat))(
            self.all_mask_idxs, tensor_flat
        )

    def compute_element_params(
        self,
        element_idx_flat: int,
        tensor_transformed: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Computes the params (regression coefficient).

        Args:
          element_idx_flat: the coordinate of the element being predict relative
            to in tensor.flatten()
          tensor_transformed: a tuple of 2 elements. One is the coefficient matrix
            and the other the target. Check the description at the top of this file
            for more details.

        Returns:
          Regression coefficients (params).
        """
        element_idx_coord = self.flat_idx_to_coordinates(element_idx_flat)

        l1_distance = self.compute_l1_distance(
            element_idx_coord, self.all_elements_idx_coord
        )
        causal_mask = self.all_elements_idx_flat < element_idx_flat

        reg = (
            self.ols_regularization_coef * 80 * jnp.eye(self.neighborhood_size) / 127.5
        )
        coef_matrix = (
            causal_mask.reshape(-1, 1, 1)
            * jnp.power(0.8, l1_distance).reshape(-1, 1, 1)
            * tensor_transformed[0]
        ).sum(axis=0) + reg

        target = (
            causal_mask.reshape(-1, 1)
            * jnp.power(0.8, l1_distance).reshape(-1, 1)
            * tensor_transformed[1]
        ).sum(axis=0)

        return jnp.linalg.pinv(coef_matrix) @ target

    def compute_tensor_params(self, tensor_transformed: jnp.ndarray) -> jnp.ndarray:
        """Computes the LASI params (regression coeff) relative to the tensor.

        Args:
          tensor_transformed: your tensor transformed with `LASI.transform_tensor`.

        Returns:
          LASI params
        """
        tensor_params = jax.vmap(
            partial(self.compute_element_params, tensor_transformed=tensor_transformed)
        )(self.all_elements_idx_flat)
        return tensor_params

    def predict_element(
        self,
        mask_idx: jnp.ndarray,
        tensor: jnp.ndarray,
        element_params: jnp.ndarray,
    ) -> float:
        """Predicts the element based on the params and neighborhood.

        Args:
          mask_idx: jax array computed from `compute_l1_mask_idxs`.
          tensor: your tensor, used to get the neighborhood.
          element_params: uses params as the regression coefficient

        Returns:
          The predicted value.
        """
        neighborhood = self.get_element_neighborhood(mask_idx, tensor.flatten())
        return element_params @ neighborhood

    def predict_tensor(
        self, tensor: jnp.ndarray, tensor_params: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Predicts the elements of the tensor based on the neighborhoods.

        Args:
          tensor: your tensor. Used to get the neighborhood.
          tensor_params: tensor parameters.

        Returns:
          A tuple of 2 elements containing the predicted tensor and variance
          estimates that are used to compute the loglikelihood with
          `compute_loglikelihood`.
        """
        if tensor_params is None:
            tensor_params = self.compute_tensor_params(self.transform_tensor(tensor))

        tensor_pred_flat = jax.vmap(partial(self.predict_element, tensor=tensor))(
            mask_idx=self.all_mask_idxs, element_params=tensor_params
        )

        return tensor_pred_flat.reshape(tensor.shape)

    def compute_distance(self, tensor_x: jnp.ndarray,  tensor_y: jnp.ndarray) -> Tuple[float, float]:
        """Computes the distance between a pair of tensors.
       
        Usage:
            >>> tensor_x, tensor_y = ... # arbitrary jax arrays of the same shape
            >>> lasi = LASI(shape, neighborhood_size)
            >>> lasi.compute_distance(tensor_x, tensor_y)
            0.3532
        """

        def _compute_tensor_params(tensor):
            params = self.compute_tensor_params(self.transform_tensor(tensor)) + 1e-6
            
            if self.unit_normalize_features:
                params = params / jnp.linalg.norm(params, ord=2, axis=1, keepdims=True)
            
            return params

        params_x = _compute_tensor_params(tensor_x)
        params_y = _compute_tensor_params(tensor_y)
        return jnp.linalg.norm(params_x - params_y, ord=2, axis=1).mean()


    def compute_distance_multiple(self, ref: jnp.ndarray,  **named_tensors) -> Tuple[float, float]:
        """Efficiently computes the distance between multiple pairs of tensors.
        Args:
            ref: all distances are computed relative to this tensor.
            named_tensors: dictionary with other tensors to be compared to `reference_tensor`.
        
        Usage:
            >>> ref, p0, p1 = ... # arbitrary jax arrays
            >>> lasi = LASI(shape, neighborhood_size)
            >>> lasi.compute_distance_multiple(ref=ref, p0=p0, p1=p1)
            {'p0': 0.1, 'p1': 0.2}
        """

        @jax.jit
        def _compute_tensor_params(tensor):
            params = self.compute_tensor_params(self.transform_tensor(tensor)) + 1e-6
            
            if self.unit_normalize_features:
                params = params / jnp.linalg.norm(params, ord=2, axis=1, keepdims=True)
            
            return params

        named_params = {
            name: _compute_tensor_params(tensor)
            for name, tensor in {'ref': ref, **named_tensors}.items()
        }
        
        @jax.jit
        def _compute_distance(tensor_x, tensor_y):
            return jnp.linalg.norm(tensor_x - tensor_y, ord=2, axis=1).mean()

        named_distances = {
            name: _compute_distance(named_params['ref'], param)
            for name, param in named_params.items() if name != 'ref'
        }

        return named_distances
