import numpy as np
from dataclasses import dataclass, fields
import numba
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_pokemon_prngs.util import rotate_left_u64


def dtype_dataclass(cls):
    """Decorator to turn a dataclass into a numpy dtype"""
    dataclass_cls = dataclass(cls)
    for field in fields(dataclass_cls):
        if hasattr(field, "__metadata__"):
            print(field.name, field.type.__metadata__)
    dataclass_cls.dtype = np.dtype(
        [
            (
                field.name,
                (
                    field.type.dtype
                    if hasattr(field.type, "dtype")
                    else (
                        field.type.__metadata__[0]
                        if hasattr(field.type, "__metadata__")
                        else field.type
                    )
                ),
            )
            for field in fields(dataclass_cls)
            if field.name != "dtype"
        ]
    )
    return dataclass_cls


def dtype_array(dtype: np.dtype, length: int) -> np.dtype:
    """Numpy dtype for an array of dtype of length"""
    return np.dtype([("", dtype, (length,))])[0]


def xoroshiro128plus_next(
    seed0: np.uint64, seed1: np.uint64
) -> tuple[np.uint64, np.uint64]:
    seed1 ^= seed0
    return np.uint64(rotate_left_u64(seed0, 24)) ^ seed1 ^ (
        seed1 << np.uint64(16)
    ), np.uint64(rotate_left_u64(seed1, 37))


def ec_pid_matrix(shiny_rolls: int) -> np.ndarray:
    """Build matrix mapping from fixed_seed high -> pid low"""
    mat = np.zeros((32, 32), np.uint8)
    for bit in range(32):
        seed0, seed1 = np.uint64(1 << (bit + 32)), np.uint64(0)
        seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # ec
        seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # tidsid
        for _ in range(shiny_rolls - 1):
            seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # pid
        for pid_bit in range(16):
            mat[bit, pid_bit] = (seed0 >> np.uint64(pid_bit)) & np.uint64(1)
            mat[bit, 16 + pid_bit] = (seed1 >> np.uint64(pid_bit)) & np.uint64(1)
    return mat


def ec_pid_const(fixed_seed_low: int, shiny_rolls: int) -> np.array:
    """Compute the xoroshiro constant & fixed_seed low's impact on pid low"""
    vec = np.zeros(32, np.uint8)
    seed0, seed1 = np.uint64(fixed_seed_low), np.uint64(0x82A2B175229D6A5B)
    seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # ec
    seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # tidsid
    for _ in range(shiny_rolls - 1):
        seed0, seed1 = xoroshiro128plus_next(seed0, seed1)  # pid
    for bit in range(16):
        vec[bit] = (seed0 >> np.uint64(bit)) & np.uint64(1)
        vec[16 + bit] = (seed1 >> np.uint64(bit)) & np.uint64(1)

    return vec


@numba.njit
def find_test_seeds_ec_pid(
    pid_low: np.uint16,
    seed_mat: np.ndarray,
    nullspace: np.ndarray,
    xoro_const: np.uint64,
):
    """Find a list of fixed_seed high values that map to the given pid low"""
    seeds = np.empty(0x10000 * len(nullspace), np.uint64)
    seed_i = 0
    for pid_s1 in range(0x10000):
        pid_s0 = (pid_low - pid_s1) & 0xFFFF
        pid_vec = (np.uint64(pid_s0) | (np.uint64(pid_s1) << 16)) ^ xoro_const

        base_seed = np.uint64(0)
        i = 0
        while pid_vec:
            if pid_vec & 1:
                base_seed ^= seed_mat[i]
            pid_vec >>= 1
            i += 1
        for nullspace_vector in nullspace:
            seeds[seed_i] = base_seed ^ nullspace_vector
            seed_i += 1
    return seeds


@numba.njit
def generate_fix_init_spec(
    fixed_seed: np.uint64,
    gender_ratio: np.uint8,
    shiny_rolls: np.uint8,
    shiny_locked: np.bool8,
    guaranteed_ivs: np.uint8,
    is_alpha: np.uint8,
    real_sidtid: np.uint32,
):
    rng = Xoroshiro128PlusRejection(
        np.uint64(fixed_seed), np.uint64(0x82A2B175229D6A5B)
    )
    encryption_constant = np.uint32(rng.next_rand(0xFFFFFFFF))
    sidtid = np.uint32(rng.next_rand(0xFFFFFFFF))
    for _ in range(1 if shiny_locked else shiny_rolls):
        pid = np.uint32(rng.next_rand(0xFFFFFFFF))
        temp = pid ^ sidtid
        xor = (temp & 0xFFFF) ^ (temp >> 16)
        shiny = xor < 16
        if shiny:
            break
    temp = pid ^ real_sidtid
    real_shiny = ((temp & 0xFFFF) ^ (temp >> 16)) < 16
    if shiny_locked and real_shiny:
        pid ^= 0x10000000
    if (not shiny_locked) and (not real_shiny) and shiny:
        pid = (pid & 0xFFFF) | (
            (
                (pid & 0xFFFF)
                ^ (real_sidtid & 0xFFFF)
                ^ (real_sidtid >> 16)
                ^ (1 if xor > 0 else 0)
            )
            << 16
        )
    ivs = np.zeros(6, np.uint8)
    for _ in range(guaranteed_ivs):
        index = rng.next_rand(6)
        while ivs[index] != 0:
            index = rng.next_rand(6)
        ivs[index] = 31
    for i in range(6):
        if ivs[i] == 0:
            ivs[i] = rng.next_rand(32)
    ability = rng.next_rand(2)
    gender = 0 if gender_ratio == 0 else 1 if gender_ratio == 254 else 2
    if 1 <= gender_ratio <= 253:
        gender = int((rng.next_rand(253) + 1) < gender_ratio)
    nature = rng.next_rand(25)
    if is_alpha:
        height = weight = 255
    else:
        height = rng.next_rand(0x81) + rng.next_rand(0x80)
        weight = rng.next_rand(0x81) + rng.next_rand(0x80)

    return shiny, encryption_constant, pid, ivs, ability, gender, nature, height, weight
