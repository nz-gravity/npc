import numpy as np

def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer))

def _is_pos_int(x) -> bool:
    return _is_int(x) and x > 0

def _is_bool(x) -> bool:
    return isinstance(x, (bool, np.bool_))

def _as_1d_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr

def _as_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr

def input_test(
    per,
    n,
    n_weights,
    burnin,
    Spar,
    degree,
    f,
    fs,
    blocked,
    data_bin_edges,
    data_bin_weights,
    log_data,
    equidistant,
    thin,
    amh,
):
    """
    Validate inputs for the MCMC.

    Raises
    ------
    ValueError
        If any input fails validation.
    """

    # Periodogram
    per = _as_array(per, "per")
    if per.ndim not in (1, 2):
        raise ValueError(f"`per` must be 1-D or 2-D, got {per.ndim}-D with shape {per.shape}.")

    if blocked:
        if per.ndim != 2:
            raise ValueError("With blocked=True, `per` must be 2-D (J x Nfreq).")
    else:
        if per.ndim != 1:
            raise ValueError("With blocked=False, `per` must be 1-D (Nfreq,).")

    nfreq = per.shape[1] if per.ndim == 2 else per.shape[0]

    # Iterations
    if not _is_pos_int(n):
        raise ValueError(f"`n` must be a positive integer, got {n}.")

    # number of coefficients
    if not _is_pos_int(n_weights):
        raise ValueError(f"`n_weights` must be a positive integer, got {n_weights}.")

    # burnin
    if not _is_pos_int(burnin):
        raise ValueError(f"`burnin` must be a positive integer, got {burnin}.")
    if burnin >= n:
        raise ValueError(f"`burnin` ({burnin}) must be < `n` ({n}).")

    # thinning factor
    if not _is_int(thin) or thin <= 0:
        raise ValueError(f"`thin` must be a positive integer > 1, got {thin}.")

    # parametric model
    Spar = np.asarray(Spar)
    if Spar.ndim != 1:
        raise ValueError(f"`Spar` must be a 1-D vector, got shape {Spar.shape}.")
    if Spar.size != nfreq:
        raise ValueError(
            f"`Spar` length must match the frequency dimension of `per` "
            f"({nfreq}), got {Spar.size}."
        )
    if not np.all(np.isfinite(Spar)):
        raise ValueError("`Spar` contains NaN or Inf.")

    # degree of the splines
    if not _is_pos_int(degree):
        raise ValueError(f"`degree` must be a positive integer > 0, got {degree}.")

    # frequencys
    if f is not None:
        f = _as_1d_array(f, "f")
        if f.size != nfreq:
            raise ValueError(
                f"`f` length must match the frequency dimension of `per` "
                f"({nfreq}), got {f.size}."
            )

    # sampling frequency
    if fs is not None:
        if not isinstance(fs, (int, float, np.integer, np.floating)):
            raise ValueError(f"`fs` must be a real number, got type {type(fs)}.")
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"`fs` must be a positive real number, got {fs}.")

    # blocked, log_data, equidistant
    if not _is_bool(blocked):
        raise ValueError(f"`blocked` must be boolean, got {type(blocked)}.")
    if not _is_bool(log_data):
        raise ValueError(f"`log_data` must be boolean, got {type(log_data)}.")
    if not _is_bool(equidistant):
        raise ValueError(f"`equidistant` must be boolean, got {type(equidistant)}.")
    if not _is_bool(amh):
        raise ValueError(f"`amh` must be boolean, got {type(amh)}.")

    # binned knots: data_bin_edges / data_bin_weights
    if (data_bin_edges is None) ^ (data_bin_weights is None):
        raise ValueError("Provide both `data_bin_edges` and `data_bin_weights`, or neither.")

    if data_bin_edges is not None and data_bin_weights is not None:
        edges = _as_1d_array(data_bin_edges, "data_bin_edges")
        weights = _as_1d_array(data_bin_weights, "data_bin_weights")

        # len(data_bin_edges) + 1 == len(data_bin_weights)
        if (len(edges) + 1) != len(weights):
            raise ValueError(
                "length of data_bin_edges is incorrect: "
                f"data_bin_edges: {len(edges)}, data_bin_weights: {len(weights)}"
            )


    return None
