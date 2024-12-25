"""Reverse from all the information stored in a pokemon to a seed"""

import numpy as np
import numba
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA, PersonalInfo8LA
import pla_reverse.pla_reverse as pla_reverse
from util.pa8 import PA8
from util import (
    ec_pid_matrix,
    ec_pid_const,
    find_test_seeds_ec_pid,
    generate_fix_init_spec,
)


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
