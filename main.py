"""Reverse from all the information stored in a pokemon to a seed"""

import numpy as np
import numba
from numba_pokemon_prngs.util import rotate_left_u64
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA, PersonalInfo8LA
import pla_reverse.pla_reverse as pla_reverse
from pa8 import PA8


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


def find_fixed_seeds_ec_pid(
    encryption_constant: np.uint32,
    pid: np.uint32,
    tsv: np.uint16,
    shiny_rolls: np.uint8,
    shiny_locked: np.bool8,
):
    fixed_seed_low = (encryption_constant - 0x229D6A5B) & 0xFFFFFFFF
    nullspace = np.array(
        tuple(
            pla_reverse.matrix.vec_to_int(row)
            for row in pla_reverse.matrix.nullspace(ec_pid_matrix(shiny_rolls))
        ),
        np.uint64,
    )
    seed_mat = np.array(
        tuple(
            pla_reverse.matrix.vec_to_int(row)
            for row in pla_reverse.matrix.generalized_inverse(
                ec_pid_matrix(shiny_rolls)
            )
        ),
        np.uint64,
    )
    xoro_const = np.uint64(
        pla_reverse.matrix.vec_to_int(ec_pid_const(fixed_seed_low, shiny_rolls))
    )

    test_seeds = np.uint64(fixed_seed_low) | (
        find_test_seeds_ec_pid(pid & 0xFFFF, seed_mat, nullspace, xoro_const)
        << np.uint64(32)
    )

    return test_seeds[
        np.where(verify_all_seeds(test_seeds, pid, tsv, shiny_rolls, shiny_locked))[0]
    ]


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


@numba.njit
def verify_all_seeds(seeds, pokemon_pid, tsv, shiny_rolls, shiny_locked):
    bools = np.zeros(seeds.shape, np.bool8)
    for i, seed in enumerate(seeds):
        seed = seeds[i]
        rng = Xoroshiro128PlusRejection(np.uint64(seed), np.uint64(0x82A2B175229D6A5B))
        rng.next_rand(0xFFFFFFFF)
        sidtid = np.uint32(rng.next_rand(0xFFFFFFFF))
        for _ in range(shiny_rolls):
            pid = np.uint32(rng.next_rand(0xFFFFFFFF))
            temp = pid ^ sidtid
            xor = (temp & 0xFFFF) ^ (temp >> 16)
            shiny = xor < 16
            if shiny:
                break
        real_shiny = (np.uint16(pid & 0xFFFF) ^ np.uint16(pid >> 16) ^ tsv) < 16
        if shiny_locked and real_shiny:
            pid ^= 0x10000000
        if (not shiny_locked) and (not real_shiny) and shiny:
            pid = (pid & 0xFFFF) | (
                ((pid & 0xFFFF) ^ tsv ^ (1 if xor > 0 else 0)) << 16
            )
        bools[i] = pid == pokemon_pid
    return bools


if __name__ == "__main__":
    SHINY_LOCKED = False
    POKEMON: PA8 = np.fromfile("pokemon.pa8", dtype=PA8.dtype).view(np.recarray)[0]
    PERSONAL_INFO: PersonalInfo8LA = PERSONAL_INFO_LA[POKEMON.species]
    if POKEMON.form:
        PERSONAL_INFO: PersonalInfo8LA = PERSONAL_INFO_LA[
            PERSONAL_INFO.form_stats_index + POKEMON.form - 1
        ]
    POKEMON_IVS = (
        POKEMON.iv32 & 0x1F,
        (POKEMON.iv32 >> 5) & 0x1F,
        (POKEMON.iv32 >> 10) & 0x1F,
        (POKEMON.iv32 >> 20) & 0x1F,
        (POKEMON.iv32 >> 25) & 0x1F,
        (POKEMON.iv32 >> 15) & 0x1F,
    )
    MAX_GUARANTEED_IVS = POKEMON_IVS.count(31)
    IS_SHINY = (
        (POKEMON.pid & 0xFFFF) ^ (POKEMON.pid >> 16) ^ POKEMON.tid ^ POKEMON.sid
    ) < 16
    IS_ALPHA = POKEMON._16 & 32

    LEGAL_SHINY_ROLLS = (
        range(1, 33)
        if IS_SHINY
        else (1, 2, 4, 5, 7, 13, 14, 16, 17, 19, 26, 27, 29, 30, 32)
    )

    for shiny_rolls in LEGAL_SHINY_ROLLS:
        seeds = find_fixed_seeds_ec_pid(
            POKEMON.encryption_constant,
            POKEMON.pid,
            POKEMON.tid ^ POKEMON.sid,
            shiny_rolls,
            SHINY_LOCKED,
        )
        for seed in seeds:
            for guaranteed_ivs in range(MAX_GUARANTEED_IVS + 1):
                (
                    shiny,
                    encryption_constant,
                    pid,
                    ivs,
                    ability,
                    gender,
                    nature,
                    height,
                    weight,
                ) = generate_fix_init_spec(
                    seed,
                    PERSONAL_INFO.gender_ratio,
                    shiny_rolls,
                    SHINY_LOCKED,
                    guaranteed_ivs,
                    IS_ALPHA,
                    POKEMON.tid | (POKEMON.sid << np.uint32(16)),
                )
                if all(iv0 == iv1 for iv0, iv1 in zip(POKEMON_IVS, ivs)):
                    print(f"Matches {guaranteed_ivs=} | {shiny_rolls=} | {seed=:016X}")
