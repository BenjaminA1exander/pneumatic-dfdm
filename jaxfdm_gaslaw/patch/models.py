from equinox import is_array

from jax.debug import print as jax_print

import jax.numpy as jnp

from jax.experimental.sparse import CSC

from jax_fdm.equilibrium.states import EquilibriumState
from jax_fdm.equilibrium.states import EquilibriumParametersState

from jax_fdm.equilibrium.sparse import sparse_solve as spsolve

from jax_fdm.equilibrium.solvers import solver_forward
from jax_fdm.equilibrium.solvers import solver_fixedpoint_implicit
from jax_fdm.equilibrium.solvers import solver_nonlinear_implicit

from jax_fdm.equilibrium.solvers import is_solver_fixedpoint
from jax_fdm.equilibrium.solvers import is_solver_leastsquares
from jax_fdm.equilibrium.solvers import is_solver_root_finding

from jax_fdm.equilibrium.loads import nodes_load_from_faces
from jax_fdm.equilibrium.loads import nodes_load_from_edges


# ==========================================================================
# Equilibrium model
# ==========================================================================

class EquilibriumModel:
    """
    A FDM model to calculate equilibrium states with shape-dependent loads.

    Parameters
    ----------
    `tmax`: The maximum number of iterations to calculate an equilibrium state.
    If `tmax=1`, the model is equivalent to doing one linear FDM step, and the rest of the
    parameters of this model are ignored. The edge and face loads are discarded too. Defaults to `100`.
    `eta`: The convergence tolerance for calculating an equilibrium state. Defaults to `1e-6`.
    `is_load_local`: If set to `True`, the face and edge loads are applied in their local coordinate system at every iteration (follower loads). Defaults to `False`.
    `itersolve_fn`: The function that calculates an equilibrium state iteratively. If `None`, the model defaults to forward fixed-point iteration.
    Note that only the solver must be consistent with the choice of residual function. Defaults to `None`.
    `iterload_fn`: A load callback that is invoked before starting iterative equilibrium computation. Defaults to `None`.
    `global_load_fn`: An optional shape-dependent load callback with signature `fn(xyz, load_state, structure) -> load_state`. If provided, it is applied inside `nodes_load` at every iteration (forward AND adjoint), so any load field it derives from the geometry (e.g. a constant-mass follower pressure `p = K / V(xyz)`) is differentiated by the implicit adjoint. Defaults to `None`.
    `implicit_diff`: If set to `True`, it applies implicit differentiation to speed up backpropagation. Defaults to `True`.
    `verbose`: Whether to print out calculation info to the terminal. Defaults to `False`.
    """
    def __init__(self,
                 tmax=100,
                 eta=1e-6,
                 is_load_local=False,
                 itersolve_fn=None,
                 iterload_fn=None,
                 implicit_diff=True,
                 verbose=False,
                 global_load_fn=None):
        self.tmax = tmax
        self.eta = eta
        self.is_load_local = is_load_local
        self.linearsolve_fn = jnp.linalg.solve
        self.itersolve_fn = itersolve_fn or solver_forward
        self.iterload_fn = iterload_fn
        self.eq_iterative_fn = self.select_equilibrium_iterative_fn(self.itersolve_fn)
        self.implicit_diff = implicit_diff
        self.verbose = verbose
        # Opt-in shape-dependent global load hook (default None => stock behaviour unchanged).
        self.global_load_fn = global_load_fn

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    @staticmethod
    def edges_vectors(xyz, connectivity):
        """
        Calculate the unnormalized edge directions (nodal coordinate differences).
        """
        return connectivity @ xyz

    @staticmethod
    def edges_lengths(vectors):
        """
        Compute the length of the edges.
        """
        return jnp.linalg.norm(vectors, axis=1, keepdims=True)

    @staticmethod
    def edges_forces(q, lengths):
        """
        Calculate the force in the edges.
        """
        return jnp.reshape(q, (-1, 1)) * lengths

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    @staticmethod
    def nodes_residuals(q, loads, vectors, connectivity):
        """
        Compute the residual forces on the nodes of the structure.
        """
        return loads - connectivity.T @ (q[:, None] * vectors)

    @staticmethod
    def nodes_positions(xyz_free, xyz_fixed, structure):
        """
        Concatenate in order the position of the free and the fixed nodes.
        """
        indices = structure.indices_freefixed

        return jnp.concatenate((xyz_free, xyz_fixed))[indices, :]

    def nodes_free_positions(self, q, xyz_fixed, loads, structure):
        """
        Calculate the XYZ coordinates of the free nodes.
        """
        K = self.stiffness_matrix(q, structure)
        P = self.load_matrix(q, xyz_fixed, loads, structure)

        return self.linearsolve_fn(K, P)

    def nodes_equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on the nodes of a structure.
        """
        return self.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)

    # --------------------------------------------------------------------
    #  Load it up!
    # --------------------------------------------------------------------

    def nodes_load(self, xyz, load_state, structure):
        """
        Calculate the loads applied to the nodes of the structure.
        """
        # Optional shape-dependent global load hook. Because this runs inside the load
        # function that both the forward fixed point and the implicit adjoint differentiate,
        # any geometry-derived load field (e.g. constant-mass pressure p = K / V(xyz)) is
        # captured exactly by the backward pass.
        if self.global_load_fn is not None:
            load_state = self.global_load_fn(xyz, load_state, structure)

        nodes_load, edges_load, faces_load = load_state

        if is_array(edges_load):
            if edges_load.size > 1:
                edges_load_ = self.edges_load(xyz, edges_load, structure, self.is_load_local)
                nodes_load = nodes_load + edges_load_

        if is_array(faces_load):
            if faces_load.size > 1:
                faces_load_ = self.faces_load(xyz, faces_load, structure, self.is_load_local)
                nodes_load = nodes_load + faces_load_

        return nodes_load

    @staticmethod
    def faces_load(xyz, faces_load, structure, is_local=False):
        """
        Calculate the tributary face loads aplied to the nodes of a structure.
        """
        return nodes_load_from_faces(xyz, faces_load, structure, is_local)

    @staticmethod
    def edges_load(xyz, edges_load, structure, is_local=False):
        """
        Calculate the tributary edge loads aplied to the nodes of a structure.
        """
        return nodes_load_from_edges(xyz, edges_load, structure, is_local)

    # ------------------------------------------------------------------------------
    #  Call me, maybe
    # ------------------------------------------------------------------------------

    def __call__(self, params, structure):
        """
        Compute an equilibrium state using the force density method (FDM).
        """
        q, xyz_fixed, load_state = params
        load_nodes = load_state.nodes

        tmax = self.tmax
        eta = self.eta
        solver = self.itersolve_fn
        iterload_fn = self.iterload_fn
        implicit_diff = self.implicit_diff
        verbose = self.verbose

        xyz_free = self.equilibrium(q, xyz_fixed, load_nodes, structure)

        if tmax > 1:

            if iterload_fn is not None:
                load_state = iterload_fn(load_state)
                params = EquilibriumParametersState(q, xyz_fixed, load_state)

            xyz_free = self.eq_iterative_fn(
                q,
                xyz_fixed,
                load_state,
                structure,
                xyz_free_init=xyz_free,
                tmax=tmax,
                eta=eta,
                solver=solver,
                implicit_diff=implicit_diff,
                verbose=verbose)

        if self.verbose:
            residuals_free = self.residual_free_matrix(params, xyz_free, structure)
            jax_print("Mean free residual vector: {}", jnp.mean(jnp.abs(residuals_free), axis=0))

        # Exit like a champ
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)
        load_nodes = self.nodes_load(xyz, load_state, structure)

        return self.equilibrium_state(q, xyz, load_nodes, structure)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(self, q, xyz_fixed, loads_nodes, structure):
        """
        Calculate static equilibrium on a structure.
        """
        return self.nodes_equilibrium(q, xyz_fixed, loads_nodes, structure)

    def equilibrium_iterative_xyz(
            self,
            q,
            xyz_fixed,
            load_state,
            structure,
            xyz_free_init=None,
            tmax=100,
            eta=1e-6,
            solver=None,
            implicit_diff=True,
            verbose=False):
        """
        Calculate static equilibrium on a structure iteratively.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit differentiation.
        """
        def loads_fn(params, xyz_free):
            """
            A closure function over a structure to calculate the load matrix.
            This matrix is the RHS of the equilibrium linear system.
            """
            return self.load_xyz_matrix_from_r_fixed(params, xyz_free, structure)

        def equilibrium_iterative_fn(params, xyz_free):
            """
            Parameters
            ----------
            params: A tuple with parameters (K, R_fixed, xyz_fixed, load_state)
            xyz_free: The 3D coordinates of the free vertices.

            Returns
            -------
            xyz_free_updated: The updated 3D coordinates of the free vertices.
            """
            # Assemble load matrix
            P = loads_fn(params, xyz_free)

            # Fetch stiffness matrix
            K = params[0]

            return self.linearsolve_fn(K, P)

        if xyz_free_init is None:
            xyz_free_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        K = self.stiffness_matrix(q, structure)
        R_fixed = self.residual_fixed_matrix(q, xyz_fixed, structure)

        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "implicit_diff": implicit_diff,
                         "verbose": verbose,
                         "loads_fn": loads_fn}

        solver_kwargs = {"solver_config": solver_config,
                         "f": equilibrium_iterative_fn,
                         "a": (K, R_fixed, xyz_fixed, load_state),
                         "x_init": xyz_free_init}

        solver = solver or self.itersolve_fn
        if implicit_diff:
            return solver_fixedpoint_implicit(solver, **solver_kwargs)

        return solver(**solver_kwargs)

    def equilibrium_iterative_residual(self,
                                       q,
                                       xyz_fixed,
                                       load_state,
                                       structure,
                                       xyz_free_init=None,
                                       tmax=100,
                                       eta=1e-6,
                                       solver=None,
                                       implicit_diff=True,
                                       verbose=False):
        """
        Calculate static equilibrium on a structure iteratively.

        Notes
        -----
        This function only supports reverse mode auto-differentiation.
        To support forward-mode, we should define a custom jvp using implicit differentiation.
        """
        def residual_fn(params, xyz_free):
            """
            The residual function of the equilibrium problem.
            """
            xyz_free = jnp.reshape(xyz_free, (-1, 3))
            residuals = self.residual_free_matrix(params, xyz_free, structure)

            return residuals.ravel()

        # Recompute xyz_free_init if not input
        if xyz_free_init is None:
            xyz_free_init = self.equilibrium(q, xyz_fixed, load_state.nodes, structure)

        # Flatten XYZ free for compatibility with the loss function
        xyz_free_init = xyz_free_init.ravel()

        # Parameterss
        params = (q, xyz_fixed, load_state)

        # Solver
        solver_config = {"tmax": tmax,
                         "eta": eta,
                         "implicit_diff": False,
                         "verbose": verbose}

        solver_kwargs = {"solver_config": solver_config,
                         "fn": residual_fn,
                         "theta": params,
                         "x_init": xyz_free_init}

        solver = solver or self.itersolve_fn

        if implicit_diff:
            xyz_free_star = solver_nonlinear_implicit(solver, **solver_kwargs)
        else:
            xyz_free_star = solver(**solver_kwargs)

        xyz_free_star = jnp.reshape(xyz_free_star, (-1, 3))

        return xyz_free_star

    def select_equilibrium_iterative_fn(self, solver):
        """
        Pick the equilibrium function that is compatible with the input solver function.
        """
        if is_solver_fixedpoint(solver):
            return self.equilibrium_iterative_xyz

        if is_solver_leastsquares(solver) or is_solver_root_finding(solver):
            return self.equilibrium_iterative_residual

        raise ValueError(f"Solver {solver} is not supported!")

    # ----------------------------------------------------------------------
    # Equilibrium state
    # ----------------------------------------------------------------------

    def equilibrium_state(self, q, xyz, loads_nodes, structure):
        """
        Assembles an equilibrium state object.
        """
        connectivity = structure.connectivity

        vectors = self.edges_vectors(xyz, connectivity)
        lengths = self.edges_lengths(vectors)
        residuals = self.nodes_residuals(q, loads_nodes, vectors, connectivity)
        forces = self.edges_forces(q, lengths)

        return EquilibriumState(xyz=xyz,
                                residuals=residuals,
                                lengths=lengths,
                                forces=forces,
                                loads=loads_nodes,
                                vectors=vectors)

    # ----------------------------------------------------------------------
    # Stiffness matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        The stiffness matrix of the structure.
        """
        c_free = structure.connectivity_free

        return c_free.T @ (q[:, None] * c_free)

    # ----------------------------------------------------------------------
    # Load matrices
    # ----------------------------------------------------------------------

    def load_matrix(self, q, xyz_fixed, load_nodes, structure):
        """
        The load matrix of the structure.
        """
        R_fixed = self.residual_fixed_matrix(q, xyz_fixed, structure)

        return load_nodes[structure.indices_free, :] - R_fixed

    def load_xyz_matrix(self, params, xyz_free, structure):
        """
        Calculate loads matrix of the free nodes of the system for shape-dependent loads.
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        return self.load_matrix(q, xyz_fixed, loads_nodes, structure)

    def load_xyz_matrix_from_r_fixed(self, params, xyz_free, structure):
        """
        Calculate loads matrix of the free nodes of the system for shape-dependent loads.
        """
        _, R_fixed, xyz_fixed, load_state = params

        # Concatenate free and fixed xyz coordinates
        xyz = self.nodes_positions(xyz_free, xyz_fixed, structure)

        # Calculate shape-dependent loads with full xyz
        loads_nodes = self.nodes_load(xyz, load_state, structure)

        # Assemble load matrix
        P = loads_nodes[structure.indices_free, :] - R_fixed

        return P

    # ----------------------------------------------------------------------
    # Residual matrices
    # ----------------------------------------------------------------------

    @staticmethod
    def residual_fixed_matrix(q, xyz_fixed, structure):
        """
        The load matrix with the contribution of the fixed nodes's residuals.
        """
        c_free = structure.connectivity_free
        c_fixed = structure.connectivity_fixed

        return c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    def residual_free_matrix(self, params, xyz_free, structure):
        """
        The residuals of the free nodes of the structure.
        """
        # Unpack parameters
        q, xyz_fixed, load_state = params

        # Calculate stiffness matrix
        K = self.stiffness_matrix(q, structure)

        # Calculate load matrix
        P = self.load_xyz_matrix(params, xyz_free, structure)

        # Residual function
        residuals = K @ xyz_free - P

        return residuals


# ==========================================================================
# Sparse equilibrium model
# ==========================================================================

class EquilibriumModelSparse(EquilibriumModel):
    """
    The equilibrium model. Sparse.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearsolve_fn = spsolve

    @staticmethod
    def stiffness_matrix(q, structure):
        """
        Computes the LHS matrix in CSC format from a vector of force densities.
        """
        # shorthands
        index_array = structure.index_array
        diag_indices = structure.diag_indices
        diags = structure.diags

        nondiags_data = -q[index_array.data - 1]
        args = (nondiags_data, index_array.indices, index_array.indptr)

        K = CSC(args, shape=index_array.shape)

        # sum of force densities for each node
        # diag_fd = diags.T @ q  # for diags as CSC matrix
        diag_fd = diags @ q  # for diags as BCSR matrix
        K.data = K.data.at[diag_indices].set(diag_fd)

        return K


# ==========================================================================
# Constant-mass (ideal-gas) follower pressure model
# ==========================================================================

class ConstantMassModelSparse(EquilibriumModelSparse):
    """
    A sparse equilibrium model with a constant-mass (ideal-gas) follower pressure.

    The face-load magnitude is recomputed at every iteration as ``p = sign * gas_k / V(xyz)``,
    where ``V`` is the volume enclosed by the surface (signed projected-area form). This is
    injected through the `global_load_fn` hook of `EquilibriumModel`, so the load function
    the forward fixed point and the implicit `custom_vjp` adjoint both differentiate now
    depends on the geometry through ``V``.

    Consequences
    ------------
    * The backward pass captures the pressure-volume feedback ``dp/dV * dV/dx`` for free,
      i.e. `value_and_grad` returns the exact constant-mass gradient in one adjoint solve --
      no outer pressure-settling loop, no manual ``dpsi/dV`` chain rule.
    * The adjoint operator ``[I - K^-1 dP/dx]`` gains the negative mass feedback term, so it
      stays non-singular at the *fixed-pressure* limit point (membrane snap-through fold),
      and the forward Picard becomes more contractive there.

    Notes
    -----
    The enclosed volume needs a triangulated face list (`faces_volume`, shape ``(n_tris, 3)``
    of vertex indices in structure/node order). `unit_faces` is the per-face unit load
    template (shape ``(n_faces, 3)``, e.g. ``[0, 0, 1]`` per face); if `None`, the incoming
    `load_state.faces` is used as the template. Use with ``is_load_local=True`` so the unit
    face load follows the face normal.

    Parameters
    ----------
    `faces_volume`: Triangulated faces used to compute the enclosed volume. Required.
    `gas_k`: The constant mass (pressure-volume product ``p * V``). Defaults to ``1.0``.
    `sign`: ``+1`` or ``-1`` to select the inflating pressure direction. Defaults to ``1.0``.
    `unit_faces`: Per-face unit load template. If `None`, uses `load_state.faces`.
    `volume_floor`: Lower clamp on ``V`` to avoid a divide-by-zero. Defaults to ``1e-9``.
    """
    def __init__(self,
                 *args,
                 faces_volume=None,
                 gas_k=1.0,
                 sign=1.0,
                 unit_faces=None,
                 volume_floor=1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if faces_volume is None:
            raise ValueError("ConstantMassModelSparse requires `faces_volume` "
                             "(triangulated faces) to compute the enclosed volume.")
        self.faces_volume = jnp.asarray(faces_volume)
        self.gas_k = gas_k
        self.sign = sign
        self.unit_faces = None if unit_faces is None else jnp.asarray(unit_faces)
        self.volume_floor = volume_floor
        # Wire the constant-mass pressure through the differentiated load function.
        self.global_load_fn = self.constant_mass_load

    def enclosed_volume(self, xyz):
        """
        Volume enclosed between the surface and the plane z=0 (signed projected-area form).
        """
        f = self.faces_volume
        vi, vj, vk = xyz[f[:, 0]], xyz[f[:, 1]], xyz[f[:, 2]]
        axy = 0.5 * ((vj[:, 0] - vi[:, 0]) * (vk[:, 1] - vi[:, 1])
                     - (vk[:, 0] - vi[:, 0]) * (vj[:, 1] - vi[:, 1]))
        zc = (vi[:, 2] + vj[:, 2] + vk[:, 2]) / 3.0

        return jnp.abs(jnp.sum(axy * zc))

    def constant_mass_load(self, xyz, load_state, structure):
        """
        Replace the face loads with the constant-mass follower pressure p = sign * gas_k / V.
        """
        volume = jnp.maximum(self.enclosed_volume(xyz), self.volume_floor)
        pressure = self.sign * self.gas_k / volume
        unit_faces = self.unit_faces if self.unit_faces is not None else load_state.faces

        return load_state._replace(faces=unit_faces * pressure)
